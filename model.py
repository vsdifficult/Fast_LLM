import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import re
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TaskSolvingLLM")

#######################################################
# Base Model Architecture Components
#######################################################

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding implementation for improved positional encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create frequencies for each dimension
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        
        # Create sin/cos angle table (max_seq_len, dim/2)
        freqs = torch.outer(positions, freqs)
        
        # Create complex exponentials e^(i*theta) = cos(theta) + i*sin(theta)
        self.cos_cached = torch.cos(freqs).view(1, max_seq_len, 1, dim // 2)
        self.sin_cached = torch.sin(freqs).view(1, max_seq_len, 1, dim // 2)
    
    def forward(self, x, seq_len: Optional[int] = None):
        # x: (batch, seq_len, heads, dim)
        seq_len = seq_len or x.shape[1]
        
        # Get cached sin/cos values up to needed sequence length
        cos = self.cos_cached[:, :seq_len, :, :].to(x.device)
        sin = self.sin_cached[:, :seq_len, :, :].to(x.device)
        
        # Split dimensions into even and odd indices
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        
        # Apply rotation (complex multiplication)
        # [x_even; x_odd] * [cos; sin] = [x_even*cos - x_odd*sin; x_even*sin + x_odd*cos]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Merge back together
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, :, 0::2] = x_rotated_even
        x_rotated[:, :, :, 1::2] = x_rotated_odd
        
        return x_rotated


class FlashAttention(nn.Module):
    """Optimized attention mechanism with support for KV caching."""
    
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
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE positional encoding
        q = self.rotary_emb(q, seq_len)
        k = self.rotary_emb(k, seq_len)
        
        # Handle KV cache for autoregressive decoding
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v)
        
        # Get effective sequence length (accounting for cache)
        effective_seq_len = k.size(1)
        
        # Transpose for attention computation: (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention weights
        # (batch, heads, seq_q, dim) @ (batch, heads, dim, seq_k) = (batch, heads, seq_q, seq_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Compute softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, dim) = (batch, heads, seq_q, dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output, new_kv_cache


class FeedForward(nn.Module):
    """Enhanced Feed-Forward layer with SwiGLU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # For SwiGLU activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU activation: swish(x*W1) * (x*W3)
        swish = self.w1(x) * torch.sigmoid(self.w1(x) * 1.0)
        gate = self.w3(x)
        x = swish * gate
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerLayer(nn.Module):
    """Optimized transformer layer with pre-normalization."""
    
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
        # Pre-normalization for attention
        normed_x = self.attn_norm(x)
        attn_output, new_kv_cache = self.attn(normed_x, kv_cache, attention_mask)
        x = x + attn_output
        
        # Pre-normalization for FF
        normed_x = self.ff_norm(x)
        ff_output = self.ff(normed_x)
        x = x + ff_output
        
        return x, new_kv_cache


class QuantizedLinear(nn.Module):
    """8-bit quantized linear layer for model efficiency."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create 8-bit quantized weights
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8))
        self.scale = nn.Parameter(torch.ones(out_features, 1, dtype=torch.float32))
        
        # Initialization
        nn_weight = torch.empty(out_features, in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(nn_weight, a=math.sqrt(5))
        
        # Quantize weights to int8
        weight_scale = nn_weight.abs().max(dim=1, keepdim=True)[0] / 127.0
        quantized_weight = (nn_weight / weight_scale).round().clamp(-127, 127).to(torch.int8)
        
        self.weight.data.copy_(quantized_weight)
        self.scale.data.copy_(weight_scale)
    
    def forward(self, x):
        # Dequantize weights to float32 for computation
        float_weight = self.weight.float() * self.scale
        return F.linear(x, float_weight)


#######################################################
# Memory and Cognitive Components
#######################################################

class MemoryItem:
    """Individual memory item with metadata."""
    
    def __init__(
        self, 
        memory_type: str, 
        content: str, 
        embedding: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.type = memory_type
        self.content = content
        self.embedding = embedding.detach().cpu() if embedding is not None else None
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            # Embedding is handled separately
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[torch.Tensor] = None) -> "MemoryItem":
        """Create memory item from dictionary."""
        memory = cls(
            memory_type=data["type"],
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {})
        )
        memory.timestamp = data.get("timestamp", datetime.now().isoformat())
        return memory


class MemoryStore:
    """Enhanced long-term memory with persistence and sophisticated retrieval."""
    
    def __init__(self, capacity: int = 10000, storage_path: Optional[str] = None):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.storage_path = storage_path
        
        # Load memories if storage path is provided
        if storage_path and os.path.exists(storage_path):
            self.load_memories()
    
    def add(
        self, 
        memory_type: str, 
        content: str, 
        embedding: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """Add an item to memory with timestamp and metadata."""
        memory = MemoryItem(memory_type, content, embedding, metadata)
        self.memories.append(memory)
        
        # Save to disk if storage path exists
        if self.storage_path:
            self.save_memories()
            
        return memory
    
    def save_memories(self):
        """Save memories to disk."""
        if not self.storage_path:
            return
            
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Save memory text content and metadata
        memory_data = [memory.to_dict() for memory in self.memories]
        
        with open(self.storage_path, 'w') as f:
            json.dump(memory_data, f)
            
        # Save embeddings separately
        if any(memory.embedding is not None for memory in self.memories):
            embeddings = [
                memory.embedding.numpy() if memory.embedding is not None 
                else np.zeros((1, 1))  # Placeholder for memories without embeddings
                for memory in self.memories
            ]
            embedding_path = self.storage_path.replace(".json", "_embeddings.npy")
            np.save(embedding_path, embeddings)
    
    def load_memories(self):
        """Load memories from disk."""
        if not os.path.exists(self.storage_path):
            return
            
        # Load memory content and metadata
        with open(self.storage_path, 'r') as f:
            memory_data = json.load(f)
        
        # Load embeddings if they exist
        embedding_path = self.storage_path.replace(".json", "_embeddings.npy")
        embeddings = None
        if os.path.exists(embedding_path):
            try:
                embeddings_array = np.load(embedding_path, allow_pickle=True)
                embeddings = [
                    torch.tensor(emb) if emb.size > 1 else None
                    for emb in embeddings_array
                ]
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        
        # Recreate memories
        self.memories = deque(maxlen=self.capacity)
        for i, data in enumerate(memory_data):
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            memory = MemoryItem.from_dict(data, emb)
            self.memories.append(memory)
    
    def search(
        self, 
        query_embedding: Optional[torch.Tensor] = None, 
        query_text: Optional[str] = None,
        memory_type: Optional[str] = None,
        k: int = 5,
        threshold: float = 0.6
    ) -> List[MemoryItem]:
        """Enhanced memory search with filtering options."""
        if len(self.memories) == 0:
            return []
        
        # First, filter by memory type if specified
        filtered_memories = list(self.memories)
        if memory_type:
            filtered_memories = [m for m in filtered_memories if m.type == memory_type]
            
        if not filtered_memories:
            return []
            
        # Search by vector similarity if embedding provided
        if query_embedding is not None:
            memories_with_embeddings = [m for m in filtered_memories if m.embedding is not None]
            
            if not memories_with_embeddings:
                return self.get_recent(k, memory_type)
                
            # Compute similarities
            embeddings = torch.stack([m.embedding for m in memories_with_embeddings])
            similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings)
            
            # Filter by threshold and get top-k
            mask = similarities >= threshold
            if not mask.any():
                return self.get_recent(k, memory_type)
                
            filtered_similarities = similarities[mask]
            filtered_indices = torch.nonzero(mask).squeeze(-1)
            
            if len(filtered_similarities) <= k:
                selected_indices = filtered_indices
            else:
                _, top_indices = torch.topk(filtered_similarities, k)
                selected_indices = filtered_indices[top_indices]
                
            return [memories_with_embeddings[i] for i in selected_indices.tolist()]
        
        # Search by text if query provided
        elif query_text:
            results = []
            
            # Tokenize query for better matching
            query_tokens = set(query_text.lower().split())
            
            # Score each memory based on token overlap
            scored_memories = []
            for memory in filtered_memories:
                memory_tokens = set(memory.content.lower().split())
                overlap = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
                if overlap > 0:
                    scored_memories.append((memory, overlap))
                    
            # Sort by score and take top k
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            results = [memory for memory, score in scored_memories[:k] if score >= threshold]
            
            return results if results else self.get_recent(k, memory_type)
        
        # Default to recent memories
        return self.get_recent(k, memory_type)
    
    def get_recent(self, k: int = 5, memory_type: Optional[str] = None) -> List[MemoryItem]:
        """Get the k most recent memories, optionally filtered by type."""
        if memory_type:
            filtered = [m for m in self.memories if m.type == memory_type]
            return list(reversed(filtered))[:k]
        else:
            return list(reversed(list(self.memories)))[:k]
    
    def clear(self):
        """Clear all memories."""
        self.memories.clear()
        
        # Remove saved files if they exist
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
                embedding_path = self.storage_path.replace(".json", "_embeddings.npy")
                if os.path.exists(embedding_path):
                    os.remove(embedding_path)
            except Exception as e:
                logger.error(f"Error removing memory files: {e}")


class WorkingMemory:
    """Enhanced short-term memory for the current thinking process."""
    
    def __init__(self, max_size: int = 15):
        self.thoughts = []
        self.max_size = max_size
        self.context = {}  # Current context information
        self.scratch_pad = ""  # Area for temporary calculations
    
    def add_thought(self, thought: str, thought_type: str = "thinking"):
        """Add a thought to working memory with type."""
        self.thoughts.append({
            "content": thought,
            "type": thought_type,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.thoughts) > self.max_size:
            self.thoughts.pop(0)
    
    def update_context(self, key: str, value: Any):
        """Update context information."""
        self.context[key] = value
    
    def update_scratch_pad(self, content: str):
        """Update or replace scratch pad content."""
        self.scratch_pad = content
    
    def append_to_scratch_pad(self, content: str):
        """Append content to scratch pad."""
        self.scratch_pad += content
    
    def clear_scratch_pad(self):
        """Clear scratch pad content."""
        self.scratch_pad = ""
    
    def get_formatted(self) -> str:
        """Get formatted working memory for prompt construction."""
        formatted = "Previous thoughts:\n"
        for i, thought in enumerate(self.thoughts, 1):
            formatted += f"{i}. [{thought['type']}] {thought['content']}\n"
            
        if self.context:
            formatted += "\nContext:\n"
            for key, value in self.context.items():
                formatted += f"- {key}: {value}\n"
                
        if self.scratch_pad:
            formatted += "\nScratch Pad (for calculations):\n"
            formatted += self.scratch_pad + "\n"
                
        return formatted
    
    def clear(self):
        """Clear working memory."""
        self.thoughts = []
        self.context = {}
        self.scratch_pad = ""


#######################################################
# Specialized Task Solving Components
#######################################################

class TaskParser:
    """Parse and understand different types of user tasks."""
    
    TASK_TYPES = {
        "question_answering": [
            r"(?:what|who|when|where|why|how|can you|could you tell|explain)",
            r"(?:tell me about|explain|describe|what is|who is|define)",
        ],
        "code_generation": [
            r"(?:write|create|generate|implement|code|program|function|class)",
            r"(?:in (?:python|javascript|java|c\+\+|ruby|go|rust|php))",
        ],
        "math_problem": [
            r"(?:calculate|compute|solve|find|evaluate|what is|determine)",
            r"(?:\d+\s*[\+\-\*\/\^\(\)]+|\bequation\b|\balgebra\b|\bintegral\b)",
        ],
        "creative_writing": [
            r"(?:write|draft|create|compose) (?:a|an|the) (?:story|poem|essay|letter|blog|article)",
            r"(?:creative|fiction|narrative|imaginative)",
        ],
        "summarization": [
            r"(?:summarize|summary|tldr|brief|overview|recap)",
            r"(?:condense|shorten|simplify)",
        ],
        "translation": [
            r"(?:translate|convert|change) (?:to|into|from) (?:english|spanish|french|german|chinese|russian|japanese|korean|italian|portuguese|arabic)",
            r"(?:translation|translator)",
        ],
        "reasoning": [
            r"(?:reason|think through|analyze|evaluate|assess|critique)",
            r"(?:logical|critical|careful|step by step)",
        ],
    }
    
    @classmethod
    def classify_task(cls, query: str) -> str:
        """Classify the task type based on the query."""
        query = query.lower()
        
        # Check each task type
        for task_type, patterns in cls.TASK_TYPES.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return task_type
        
        # Default to question answering
        return "question_answering"
    
    @classmethod
    def extract_constraints(cls, query: str) -> Dict[str, Any]:
        """Extract constraints and parameters from the query."""
        constraints = {}
        
        # Look for length constraints
        length_match = re.search(r"(?:in|within|at most|at least|about) (\d+) (?:words|sentences|paragraphs|lines)", query.lower())
        if length_match:
            constraints["length"] = int(length_match.group(1))
            constraints["length_unit"] = re.search(r"(?:words|sentences|paragraphs|lines)", length_match.group(0)).group(0)
        
        # Look for format constraints
        format_patterns = [
            (r"as (?:a|an) (list|table|diagram|chart|graph|json|xml|markdown|outline)", "format"),
            (r"in (?:a|an) (formal|informal|academic|professional|casual|conversational) (?:style|tone|manner)", "tone"),
            (r"for (?:a|an) (beginner|intermediate|advanced|expert|technical|non-technical|general) audience", "audience"),
        ]
        
        for pattern, key in format_patterns:
            match = re.search(pattern, query.lower())
            if match:
                constraints[key] = match.group(1)
        
        # For code generation tasks, extract language
        if cls.classify_task(query) == "code_generation":
            lang_match = re.search(r"in (python|javascript|java|c\+\+|ruby|go|rust|php|html|css|sql|bash|typescript)", query.lower())
            if lang_match:
                constraints["language"] = lang_match.group(1)
        
        return constraints


class ReasoningEngine:
    """Enhanced engine for step-by-step reasoning and reflection."""
    
    def __init__(self, model, tokenizer, max_steps: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.working_memory = WorkingMemory()
        self.verification_enabled = True
    
    def think(self, query: str, context: str = "", max_tokens_per_step: int = 150) -> Dict:
        """Perform multi-step thinking with different reasoning strategies."""
        self.working_memory.clear()
        
        # Determine reasoning strategy based on task
        task_type = TaskParser.classify_task(query)
        
        if task_type == "math_problem":
            return self._mathematical_reasoning(query, context, max_tokens_per_step)
        elif task_type == "code_generation":
            return self._code_reasoning(query, context, max_tokens_per_step)
        elif task_type in ["reasoning", "question_answering"]:
            return self._analytical_reasoning(query, context, max_tokens_per_step)
        else:
            # Default reasoning approach
            return self._default_reasoning(query, context, max_tokens_per_step)
    
    def _build_prompt(self, query: str, context: str, instruction: str) -> str:
        """Build a prompt with query, context, and working memory."""
        prompt = f"Question: {query}\n\n"
        
        if context:
            prompt += f"Context:\n{context}\n\n"
            
        prompt += f"{self.working_memory.get_formatted()}\n"
        prompt += f"{instruction}\n"
        
        return prompt
    
    def _generate_thought(self, prompt: str, max_tokens: int) -> str:
        """Generate a thought using the model."""
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
            )
        
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
    
    def _default_reasoning(self, query: str, context: str, max_tokens_per_step: int) -> Dict:
        """Standard reasoning approach for general questions."""
        self.working_memory.add_thought(f"Initial question: {query}", "question")
        
        all_thoughts = []
        current_step = 1
        
        # Multi-step thinking loop
        for _ in range(self.max_steps):
            # Generate next thought
            instruction = f"Thinking step {current_step}: I'll reason through this carefully..."
            prompt = self._build_prompt(query, context, instruction)
            
            thought = self._generate_thought(prompt, max_tokens_per_step)
            
            # Save the thought
            step_thought = f"Step {current_step}: {thought}"
            all_thoughts.append(step_thought)
            self.working_memory.add_thought(thought, "thinking")
            
            # Check if we have a conclusion
            if any(phrase in thought.lower() for phrase in 
                   ["therefore", "conclusion", "answer is", "in summary", "finally"]):
                break
                
            current_step += 1
        
        # Generate final answer if needed
        if not any(phrase in " ".join(all_thoughts).lower() for phrase in 
                  ["therefore", "conclusion", "answer is", "in summary", "finally"]):
            
            instruction = "Based on my analysis above, my final answer is:"
            prompt = self._build_prompt(query, context, instruction)
            
            final_answer = self._generate_thought(prompt, max_tokens_per_step)
            
            all_thoughts.append(f"Conclusion: {final_answer}")
            self.working_memory.add_thought(f"Final answer: {final_answer}", "conclusion")
        
        return {
            "steps": all_thoughts,
            "final_answer": all_thoughts[-1],
            "working_memory": self.working_memory.get_formatted()
        }
    
    def _mathematical_reasoning(self, query: str, context: str, max_tokens_per_step: int) -> Dict:
        """Specialized reasoning for mathematical problems."""
        self.working_memory.add_thought(f"Mathematical problem: {query}", "question")
        
        all_thoughts = []
        current_step = 1
        
        # First, understand the problem
        instruction = "Let me understand this math problem by identifying the key variables and what I need to find:"
        prompt = self._build_prompt(query, context, instruction)
        
        understanding = self._generate_thought(prompt, max_tokens_per_step)
        all_thoughts.append(f"Understanding: {understanding}")
        self.working_memory.add_thought(understanding, "understanding")
        
        # Multi-step solving
        for _ in range(self.max_steps - 1):  # -1 because we used one step for understanding
            # Determine the next step based on current progress
            if current_step == 1:
                instruction = "Let me start solving this step-by-step:"
            else:
                instruction = f"Step {current_step} of the solution:"
            
            prompt = self._build_prompt(query, context, instruction)
            thought = self._generate_thought(prompt, max_tokens_per_step)
            
            step_thought = f"Step {current_step}: {thought}"
            all_thoughts.append(step_thought)
            self.working_memory.add_thought(thought, "calculation")
            
            # Check if we've reached the solution
            if any(phrase in thought.lower() for phrase in 
                   ["therefore", "thus", "so", "answer is", "result is", "solution is", "=", "equals"]):
                break
                
            current_step += 1
        
        # Verify the solution if enabled
        if self.verification_enabled:
            instruction = "Let me verify my solution by checking my work:"
            prompt = self._build_prompt(query, context, instruction)
            verification = self._generate_thought(prompt, max_tokens_per_step)
            
            all_thoughts.append(f"Verification: {verification}")
            self.working_memory.add_thought(verification, "verification")
        
        # Final answer
        instruction = "Based on my calculations, the final answer is:"
        prompt = self._build_prompt(query, context, instruction)
        final_answer = self._generate_thought(prompt, max_tokens_per_step)
        
        all_thoughts.append(f"Answer: {final_answer}")
        self.working_memory.add_thought(final_answer, "answer")
        
        return {
            "steps": all_thoughts,
            "final_answer": final_answer,
            "working_memory": self.working_memory.get_formatted()
        }
    
    def _code_reasoning(self, query: str, context: str, max_tokens_per_step: int) -> Dict:
        """Specialized reasoning for code generation tasks."""
        self.working_memory.add_thought(f"Code task: {query}", "question")
        
        all_thoughts = []
        
        # Understand requirements
        instruction = "Let me analyze the requirements for this code task:"
        prompt = self._build_prompt(query, context, instruction)
        requirements = self._generate_thought(prompt, max_tokens_per_step)
        
        all_thoughts.append(f"Requirements: {requirements}")
        self.working_memory.add_thought(requirements, "requirements")
        
        # Design approach
        instruction = "Let me design a solution approach with pseudocode or high-level description:"
        prompt = self._build_prompt(query, context, instruction)
        design = self._generate_thought(prompt, max_tokens_per_step)
        
        all_thoughts.append(f"Design: {design}")
        self.working_memory.add_thought(design, "design")
        
        # Generate code
        constraints = TaskParser.extract_constraints(query)
        language = constraints.get("language", "python")
        
        instruction = f"Now I'll implement the solution in {language}:"
        prompt = self._build_prompt(query, context, instruction)
        
        # Use more tokens for code generation
        code = self._generate_thought(prompt, max_tokens_per_step * 2)
        
        all_thoughts.append(f"Implementation: {code}")
        self.working_memory.add_thought(code, "implementation")
        
        # Test cases or example usage
        instruction = "Let me provide test cases or example usage to demonstrate the code:"
        prompt = self._build_prompt(query, context, instruction)
        testing = self._generate_thought(prompt, max_tokens_per_step)
        
        all_thoughts.append(f"Testing: {testing}")
        self.working_memory.add_thought(testing, "testing")
        
        # Format final solution
        instruction = "Let me present the complete solution with explanation:"
        prompt = self._build_prompt(query, context, instruction)
        final_solution = self._generate_thought(prompt, max_tokens_per_step * 2)
        
        all_thoughts.append(f"Solution: {final_solution}")
        self.working_memory.add_thought(final_solution, "solution")
        
        return {
            "steps": all_thoughts,
            "final_answer": final_solution,
            "working_memory": self.working_memory.get_formatted()
        }
    
    def _analytical_reasoning(self, query: str, context: str, max_tokens_per_step: int) -> Dict:
        """Critical analytical reasoning for complex questions."""
        self.working_memory.add_thought(f"Analysis question: {query}", "question")
        
        all_thoughts = []
        
        # Define perspectives/angles to consider
        perspectives = [
            "Let me first consider the key facts and definitions:",
            "Let me analyze the underlying assumptions:",
            "Let me consider different viewpoints and perspectives:",
            "Let me identify potential logical fallacies or biases:",
            "Let me examine the evidence and support for each position:",
            "Let me synthesize the information and draw connections:"
        ]
        
        # Generate thoughts from different perspectives
        for i, perspective in enumerate(perspectives[:min(len(perspectives), self.max_steps - 1)]):
            prompt = self._build_prompt(query, context, perspective)
            thought = self._generate_thought(prompt, max_tokens_per_step)
            
            angle = perspective.replace("Let me ", "").replace(":", "")
            all_thoughts.append(f"{angle}: {thought}")
            self.working_memory.add_thought(thought, f"perspective_{i+1}")
        
        # Generate conclusion
        instruction = "Based on my comprehensive analysis, I can now conclude:"
        prompt = self._build_prompt(query, context, instruction)
        conclusion = self._generate_thought(prompt, max_tokens_per_step)
        
        all_thoughts.append(f"Conclusion: {conclusion}")
        self.working_memory.add_thought(conclusion, "conclusion")
        
        return {
            "steps": all_thoughts,
            "final_answer": conclusion,
            "working_memory": self.working_memory.get_formatted()
        }


#######################################################
# Advanced Tokenization
#######################################################

class AdvancedTokenizer:
    """Enhanced tokenizer with better handling of text structures."""
    
    def __init__(self, vocab_size=50257, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        
        # Token IDs
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        
        # Initialize basic vocabulary (simplified)
        self._init_vocab()
    
    def _init_vocab(self):
        """Initialize the vocabulary with basic tokens."""
        # Special tokens
        self.token_to_id = {
            self.unk_token: self.unk_token_id,
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }
        
        # Simple vocab - would be replaced with real tokenization in production
        # Here we just create a basic character-level tokenization for demonstration
        for i, c in enumerate(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:\"'()[]{}-_+=*/\\<>@#$%^&|~`"), 3):
            self.token_to_id[c] = i
            
        # Words for common programming languages and concepts (simplified)
        common_words = ["function", "class", "def", "var", "let", "const", "import", "from", "return", 
                         "if", "else", "for", "while", "in", "of", "try", "except", "finally", "with", 
                         "async", "await", "python", "javascript", "java", "c++", "algorithm", "data", 
                         "analysis", "model", "train", "predict", "tensor", "matrix", "vector"]
                         
        word_id = max(self.token_to_id.values()) + 1
        for word in common_words:
            if word not in self.token_to_id:
                self.token_to_id[word] = word_id
                word_id += 1
        
        # Create the reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """Convert text to token IDs using a simple approach."""
        if not text:
            token_ids = [self.pad_token_id]
        else:
            # Simple tokenization (character-level with some basic words)
            # In a real tokenizer, this would use BPE, WordPiece, or other algorithm
            token_ids = []
            i = 0
            while i < len(text):
                # Try to match longer words first
                matched = False
                for word_len in range(20, 0, -1):  # Try words up to 20 chars
                    if i + word_len <= len(text):
                        word = text[i:i+word_len]
                        if word in self.token_to_id:
                            token_ids.append(self.token_to_id[word])
                            i += word_len
                            matched = True
                            break
                
                # If no word match, add character
                if not matched:
                    char = text[i]
                    token_ids.append(self.token_to_id.get(char, self.unk_token_id))
                    i += 1
            
            # Add EOS token
            token_ids.append(self.eos_token_id)
        
        # Return as tensor if requested
        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        return token_ids
    
    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        """Convert token IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        if skip_special_tokens:
            ids = [id for id in ids if id not in [self.pad_token_id, self.eos_token_id]]
        
        return "".join(self.id_to_token.get(id, self.unk_token) for id in ids)
    
    def batch_encode(self, texts: List[str], padding: bool = True, return_tensors: Optional[str] = None) -> Dict:
        """Encode a batch of texts."""
        encoded = [self.encode(text) for text in texts]
        
        # Add padding if requested
        if padding:
            max_len = max(len(ids) for ids in encoded)
            encoded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
        
        # Create attention masks
        attention_masks = [[1] * len(ids) if id != self.pad_token_id else 0 for ids in encoded for id in ids]
        
        # Return as tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
            }
        
        return {
            "input_ids": encoded,
            "attention_mask": attention_masks
        }


#######################################################
# Main TaskSolvingLLM Class
#######################################################

class TaskSolvingLLM(nn.Module):
    """Enhanced language model with task understanding and solving capabilities."""
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        use_quantization: bool = True,
        memory_capacity: int = 10000,
        memory_path: Optional[str] = None,
        tokenizer = None,
    ):
        super().__init__()
        
        # Architecture parameters
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_quantization = use_quantization
        
        # Initialize tokenizer
        self.tokenizer = tokenizer if tokenizer else AdvancedTokenizer(vocab_size)
        
        # Token embeddings
        if use_quantization:
            # Use sharding for embeddings (split into multiple smaller tables)
            self.embedding_shards = nn.ModuleList([
                nn.Embedding(min(50000, vocab_size - i*50000), dim)
                for i in range((vocab_size + 49999) // 50000)
            ])
        else:
            self.embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization and projection
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        
        if use_quantization:
            self.lm_head = QuantizedLinear(dim, vocab_size)
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Share weights between embedding and projection if not quantized
        if not use_quantization:
            self.lm_head.weight = self.embedding.weight
        
        # Memory systems
        self.long_term_memory = MemoryStore(capacity=memory_capacity, storage_path=memory_path)
        
        # Reasoning engine for tasks
        self.reasoning_engine = ReasoningEngine(self, self.tokenizer)
        
        # Additional components for task solving
        self.memory_projection = nn.Linear(dim, dim, bias=False)
        
        # Task specialization
        self.task_understanding = True
        self.self_improvement = True
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Settings
        self.verbose = False  # For debugging
    
    def _init_weights(self, module):
        """Initialize model weights."""
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
        """Get embeddings with support for sharded embedding tables."""
        if self.use_quantization:
            # Use sharding for embeddings
            embeds = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.dim), 
                device=input_ids.device, 
                dtype=torch.float32
            )
            
            for i, shard in enumerate(self.embedding_shards):
                # Create mask for tokens in this shard
                mask = (input_ids >= i*50000) & (input_ids < (i+1)*50000)
                if not mask.any():
                    continue
                
                # Get indices relative to current shard
                shard_indices = (input_ids - i*50000) * mask
                
                # Get embeddings and add them to result
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
        """Forward pass through the model."""
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.get_embeddings(input_ids)
        
        # Create new KV caches if not provided
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        
        new_kv_caches = []
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x, new_kv_cache = layer(x, kv_caches[i], attention_mask)
            new_kv_caches.append(new_kv_cache)
        
        # Apply normalization and project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_kv_caches
    
    def get_embedding_vector(self, text: str) -> torch.Tensor:
        """Get embedding vector for a text for memory operations."""
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.inference_mode():
            # Get embeddings from the model
            hidden_states = self.get_embeddings(input_ids)
            
            # Use mean pooling for sentence embedding
            mask = torch.ones_like(input_ids).float().unsqueeze(-1)
            mean_embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            
            # Project to get final embedding
            return self.memory_projection(mean_embedding).squeeze(0)
    
    def remember(self, memory_type: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Store a memory with its embedding vector."""
        embedding = self.get_embedding_vector(text)
        return self.long_term_memory.add(memory_type, text, embedding, metadata)
    
    def recall(self, query: str, k: int = 5, memory_type: Optional[str] = None) -> List[MemoryItem]:
        """Retrieve relevant memories based on a query."""
        query_embedding = self.get_embedding_vector(query)
        return self.long_term_memory.search(query_embedding=query_embedding, query_text=query, memory_type=memory_type, k=k)
    
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
        """Generate text with autoregressive decoding."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create attention mask
        seq_len = input_ids.shape[1]
        attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Initialize KV caches
        kv_caches = [None] * len(self.layers)
        
        # Prepare result and token counter
        generated_ids = input_ids.clone()
        
        # Run model on full sequence
        _, kv_caches = self.forward(input_ids, kv_caches, attention_mask)
        
        # Generate new tokens autoregressively
        for _ in range(max_new_tokens):
            # Get latest tokens
            input_ids = generated_ids[:, -1:]
            
            # Get logits for last token only
            logits, kv_caches = self.forward(input_ids, kv_caches, None)
            logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to remove first token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create index mask for removal
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Add new token to result
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated_ids
    
    def think(self, query: str) -> Dict:
        """Use multi-step reasoning for complex queries."""
        # Get relevant memories for context
        memories = self.recall(query)
        memory_context = ""
        
        if memories:
            memory_context = "Relevant knowledge:\n"
            for i, memory in enumerate(memories, 1):
                memory_context += f"{i}. {memory.content}\n"
        
        # Perform thinking
        thinking_result = self.reasoning_engine.think(query, memory_context)
        
        # Store the thinking process
        thinking_summary = "\n".join(thinking_result["steps"])
        self.remember("thinking", f"Question: {query}\nThinking process:\n{thinking_summary}")
        
        return thinking_result
    
    def solve_task(
        self,
        query: str,
        thinking_mode: Optional[bool] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Solve a user task with appropriate processing."""
        # Parse task type and constraints
        task_type = TaskParser.classify_task(query)
        constraints = TaskParser.extract_constraints(query)
        
        # Determine if thinking mode should be used
        if thinking_mode is None:
            # Use thinking mode for complex questions and reasoning tasks
            thinking_mode = task_type in ["reasoning", "math_problem", "code_generation"] or \
                            any(word in query.lower() for word in 
                               ["why", "how", "explain", "analyze", "think", "reason", "compare", "evaluate"])
        
        # Get relevant memories
        memories = self.recall(query)
        memory_context = ""
        
        if memories:
            memory_context = "Relevant previous knowledge:\n"
            for i, memory in enumerate(memories, 1):
                memory_context += f"{i}. {memory.content}\n"
        
        # Process based on task type and thinking mode
        if thinking_mode:
            # Use reasoning engine for complex tasks
            thinking_result = self.reasoning_engine.think(query, memory_context)
            answer = thinking_result["final_answer"]
            
            # Store the interaction
            self.remember("task_solution", 
                         f"Task: {query}\nSolution approach:\n{thinking_result['working_memory']}\nFinal answer:\n{answer}",
                         {"task_type": task_type})
            
            return {
                "response": answer,
                "thinking_process": thinking_result["steps"],
                "task_type": task_type,
                "memories_used": [m.content for m in memories],
            }
        else:
            # Direct generation for simpler tasks
            device = next(self.parameters()).device
            
            # Create prompt with task context
            full_prompt = f"{memory_context}User query: {query}\n\nResponse:"
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(device)
            
            # Generate response
            generated_ids = self.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # Decode response
            response = self.tokenizer.decode(
                generated_ids[0, input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Store the interaction
            self.remember("task_solution", 
                         f"Task: {query}\nSolution:\n{response}",
                         {"task_type": task_type})
            
            return {
                "response": response,
                "task_type": task_type,
                "memories_used": [m.content for m in memories],
            }
    
    def interactive_session(self):
        """Run an interactive session with the model."""
        print("TaskSolvingLLM initialized! Type 'exit' to quit.")
        print("Special commands:")
        print("  /clear - Clear working memory")
        print("  /think - Force thinking mode for next query")
        print("  /direct - Force direct generation for next query")
        print("  /memories - Show recent memories")
        print("  /save - Save all memories")
        
        thinking_mode = None  # Auto-detect
        
        while True:
            user_input = input("\nYou: ")
            
            # Handle special commands
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "/clear":
                self.reasoning_engine.working_memory.clear()
                print("Working memory cleared.")
                continue
            elif user_input.lower() == "/think":
                thinking_mode = True
                print("Thinking mode enabled for next query.")
                continue
            elif user_input.lower() == "/direct":
                thinking_mode = False
                print("Direct generation mode enabled for next query.")
                continue
            elif user_input.lower() == "/memories":
                memories = self.long_term_memory.get_recent(5)
                print("\nRecent memories:")
                for i, memory in enumerate(memories, 1):
                    print(f"{i}. [{memory.type}] {memory.content[:100]}...")
                continue
            elif user_input.lower() == "/save":
                if hasattr(self.long_term_memory, "save_memories"):
                    self.long_term_memory.save_memories()
                    print("Memories saved.")
                else:
                    print("Memory saving not enabled.")
                continue
            
            # Process query
            print("\nProcessing...")
            try:
                response = self.solve_task(user_input, thinking_mode=thinking_mode)
                
                # Reset thinking mode to auto
                thinking_mode = None
                
                # Display response
                if "thinking_process" in response:
                    print("\nThinking process:")
                    for step in response["thinking_process"]:
                        print(f"- {step}")
                        
                print(f"\nResponse: {response['response']}")
                
                # Show task classification
                print(f"\nTask identified as: {response['task_type']}")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Model parameters
    vocab_size = 50257  # GPT-2 vocabulary size
    dim = 768         # Embedding dimension
    num_layers = 12    # Number of transformer layers
    num_heads = 12     # Number of attention heads
    
    # Create tokenizer
    tokenizer = AdvancedTokenizer(vocab_size)
    
    # Create model
    model = TaskSolvingLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_quantization=True,
        tokenizer=tokenizer,
        memory_path="llm_memories.json"
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Run interactive session
    model.interactive_session()