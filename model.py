import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RotaryPositionalEmbedding(nn.Module):
    """Реализация RoPE (Rotary Position Embedding) для улучшения обработки позиционной информации."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Создаем частоты для каждой размерности
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        
        # Создаем таблицу углов sin/cos (max_seq_len, dim/2)
        freqs = torch.outer(positions, freqs)
        
        # Создаем комплексные экспоненты e^(i*theta) = cos(theta) + i*sin(theta)
        self.cos_cached = torch.cos(freqs).view(1, max_seq_len, 1, dim // 2)
        self.sin_cached = torch.sin(freqs).view(1, max_seq_len, 1, dim // 2)
    
    def forward(self, x, seq_len: Optional[int] = None):
        # x: (batch, seq_len, heads, dim)
        seq_len = seq_len or x.shape[1]
        
        # Получаем кэшированные значения sin/cos до нужной длины последовательности
        cos = self.cos_cached[:, :seq_len, :, :].to(x.device)
        sin = self.sin_cached[:, :seq_len, :, :].to(x.device)
        
        # Разделяем размерности на четные и нечетные индексы
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        
        # Применяем вращение (rotate by complex multiplication)
        # [x_even; x_odd] * [cos; sin] = [x_even*cos - x_odd*sin; x_even*sin + x_odd*cos]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Объединяем обратно
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, :, 0::2] = x_rotated_even
        x_rotated[:, :, :, 1::2] = x_rotated_odd
        
        return x_rotated


class FlashAttention(nn.Module):
    """Оптимизированная версия механизма внимания для ускорения LLM."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len, _ = x.shape
        
        # Проекция запросов, ключей и значений
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Применяем позиционное кодирование RoPE
        q = self.rotary_emb(q, seq_len)
        k = self.rotary_emb(k, seq_len)
        
        # Обработка KV кэша для авторегрессивного декодирования
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v)
        
        # Определим эффективный размер последовательности (с учетом кэша)
        effective_seq_len = k.size(1)
        
        # Перестановка для вычисления внимания: (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Вычисление матрицы весов внимания
        # (batch, heads, seq_q, dim) @ (batch, heads, dim, seq_k) = (batch, heads, seq_q, seq_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Применяем маску внимания если она предоставлена
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Вычисляем softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Вычисляем взвешенную сумму значений
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, dim) = (batch, heads, seq_q, dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Возвращаем в исходную форму и проецируем обратно
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output, new_kv_cache


class FeedForward(nn.Module):
    """Ускоренный Feed-Forward слой с SwiGLU активацией."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Для SwiGLU активации
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU активация: swish(x*W1) * (x*W3)
        swish = self.w1(x) * torch.sigmoid(self.w1(x) * 1.0)
        gate = self.w3(x)
        x = swish * gate
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerLayer(nn.Module):
    """Оптимизированный слой трансформера с пре-нормализацией."""
    
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-5)
        self.attn = FlashAttention(dim, num_heads, dropout)
        
        self.ff_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, ff_dim, dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # Пре-нормализация для внимания
        normed_x = self.attn_norm(x)
        attn_output, new_kv_cache = self.attn(normed_x, kv_cache, attention_mask)
        x = x + attn_output
        
        # Пре-нормализация для FF
        normed_x = self.ff_norm(x)
        ff_output = self.ff(normed_x)
        x = x + ff_output
        
        return x, new_kv_cache


class QuantizedLinear(nn.Module):
    """8-битное квантование для линейных слоев."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Создаем 8-битные квантованные веса
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8))
        self.scale = nn.Parameter(torch.ones(out_features, 1, dtype=torch.float32))
        
        # Инициализация
        nn_weight = torch.empty(out_features, in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(nn_weight, a=math.sqrt(5))
        
        # Квантуем веса до int8
        weight_scale = nn_weight.abs().max(dim=1, keepdim=True)[0] / 127.0
        quantized_weight = (nn_weight / weight_scale).round().clamp(-127, 127).to(torch.int8)
        
        self.weight.data.copy_(quantized_weight)
        self.scale.data.copy_(weight_scale)
    
    def forward(self, x):
        # Деквантуем веса обратно к float32 для вычислений
        float_weight = self.weight.float() * self.scale
        return F.linear(x, float_weight)


class AcceleratedLLM(nn.Module):
    """Ускоренная LLM модель с оптимизациями для инференса."""
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_quantization: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_quantization = use_quantization
        
        # Токенные эмбеддинги
        if use_quantization:
            # Используем шардинг для эмбеддингов (разделяем на несколько меньших таблиц)
            self.embedding_shards = nn.ModuleList([
                nn.Embedding(min(50000, vocab_size - i*50000), dim)
                for i in range((vocab_size + 49999) // 50000)
            ])
        else:
            self.embedding = nn.Embedding(vocab_size, dim)
        
        # Слои трансформера
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Нормализация выхода и проекция на словарь
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        
        if use_quantization:
            self.lm_head = QuantizedLinear(dim, vocab_size)
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Разделяем веса проекции и эмбеддингов
        if not use_quantization:
            self.lm_head.weight = self.embedding.weight
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_embeddings(self, input_ids):
        if self.use_quantization:
            # Используем шардинг для эмбеддингов
            embeds = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.dim), 
                device=input_ids.device, 
                dtype=torch.float32
            )
            
            for i, shard in enumerate(self.embedding_shards):
                # Создаем маску для токенов в этом шарде
                mask = (input_ids >= i*50000) & (input_ids < (i+1)*50000)
                if not mask.any():
                    continue
                
                # Получаем индексы относительно текущего шарда
                shard_indices = (input_ids - i*50000) * mask
                
                # Получаем эмбеддинги и добавляем их к результату
                shard_embeds = shard(shard_indices)
                embeds += shard_embeds * mask.unsqueeze(-1)
            
            return embeds
        else:
            return self.embedding(input_ids)
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        kv_caches: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape
        
        # Получаем эмбеддинги токенов
        x = self.get_embeddings(input_ids)
        
        # Создаем новые KV кэши если они не предоставлены
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        
        new_kv_caches = []
        
        # Проходим через слои трансформера
        for i, layer in enumerate(self.layers):
            x, new_kv_cache = layer(x, kv_caches[i], attention_mask)
            new_kv_caches.append(new_kv_cache)
        
        # Применяем нормализацию и проецируем на словарь
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_kv_caches
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ):
        """Генерация текста с использованием авторегрессивного декодирования."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Создаем маску внимания
        seq_len = input_ids.shape[1]
        attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Инициализируем KV кэши
        kv_caches = [None] * len(self.layers)
        
        # Подготавливаем результат и счетчик токенов
        generated_ids = input_ids.clone()
        
        # Запускаем модель на всей последовательности
        _, kv_caches = self.forward(input_ids, kv_caches, attention_mask)
        
        # Генерируем новые токены авторегрессивно
        for _ in range(max_new_tokens):
            # Получаем последние токены
            input_ids = generated_ids[:, -1:]
            
            # Получаем логиты только для последнего токена
            logits, kv_caches = self.forward(input_ids, kv_caches, None)
            logits = logits[:, -1, :]
            
            # Применяем температуру
            if temperature > 0:
                logits = logits / temperature
            
            # Применяем штраф за повторения
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty
            
            # Применяем top-k фильтрацию
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Применяем top-p (nucleus) сэмплирование
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Удаляем токены с кумулятивной вероятностью выше порога
                sorted_indices_to_remove = cumulative_probs > top_p
                # Сдвигаем индексы для удаления первого токена
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Создаем маску индексов для удаления
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Сэмплируем следующий токен
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Добавляем новый токен к результату
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Проверяем на конец последовательности
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated_ids


# Пример использования
if __name__ == "__main__":
    # Параметры модели
    vocab_size = 50257  # Словарь GPT-2
    dim = 768          # Размерность эмбеддингов
    num_layers = 12    # Количество слоев
    num_heads = 12     # Количество голов внимания
    
    # Создаем модель
    model = AcceleratedLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_quantization=True,
    )
    
    # Переключаем в режим инференса
    model.eval()
    
    # Создаем входные данные
    input_ids = torch.randint(0, vocab_size, (1, 32))
    
    # Генерируем текст
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )
    
    print(f"Generated sequence length: {generated_ids.shape[1]}")
    
    # Для реального использования потребуется токенизатор
    # из библиотеки transformers для декодирования результата