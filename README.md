# TaskSolvingLLM

A powerful language model implementation with enhanced reasoning, memory, and task-solving capabilities. This project provides a complete implementation of a transformer-based language model that can understand user queries, solve various tasks, and remember past interactions.

## Features

- **Advanced Reasoning**: Multi-step thinking process with specialized strategies for math, coding, and analytical problems
- **Memory Systems**: Long-term and working memory to store and recall previous interactions and knowledge
- **Task Understanding**: Automatic classification of user queries into task types and extraction of constraints
- **Optimized Architecture**: Enhanced transformer architecture with rotary positional embeddings, flash attention, and quantization
- **Cognitive Capabilities**: Step-by-step reasoning, verification, and self-improvement mechanisms

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- numpy
- (Optional) CUDA-capable GPU for faster inference

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fast_LLM.git
   cd Fast_LLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download pre-trained weights:
   ```bash
   python download_weights.py
   ```

## Usage

### Basic Usage

```python
from task_solving_llm import TaskSolvingLLM, AdvancedTokenizer

# Initialize model and tokenizer
vocab_size = 50257  # GPT-2 vocabulary size
tokenizer = AdvancedTokenizer(vocab_size)

model = TaskSolvingLLM(
    vocab_size=vocab_size,
    dim=768,
    num_layers=12,
    num_heads=12,
    use_quantization=True,
    tokenizer=tokenizer,
    memory_path="llm_memories.json"
)

# Solve a task
result = model.solve_task("Calculate the area of a triangle with base 6 and height 8")
print(result["response"])

# Generate with step-by-step thinking
thinking_result = model.think("Explain how neural networks learn")
for step in thinking_result["steps"]:
    print(step)
print(thinking_result["final_answer"])
```

### Interactive Mode

Run the model in interactive mode to chat with it:

```python
model = TaskSolvingLLM(...)
model.interactive_session()
```

Special commands in interactive mode:
- `/clear` - Clear working memory
- `/think` - Force thinking mode for next query
- `/direct` - Force direct generation for next query
- `/memories` - Show recent memories
- `/save` - Save all memories

## Architecture

The TaskSolvingLLM consists of several key components:

### Base Model

- **Transformer Architecture**: Standard transformer with enhancements
- **Rotary Positional Embeddings**: Better handling of positional information
- **Flash Attention**: Optimized attention mechanism
- **Quantization**: Optional 8-bit quantization for efficiency

### Memory Systems

- **MemoryStore**: Long-term storage of information with vector embeddings
- **WorkingMemory**: Short-term context for reasoning tasks
- **Persistence**: Optional saving/loading of memories to/from disk

### Reasoning Engine

- **Multi-step Reasoning**: Breaking complex problems into steps
- **Task-Specific Strategies**:
  - Mathematical reasoning for calculations and equations
  - Code reasoning for programming tasks
  - Analytical reasoning for complex questions
  - Default reasoning for general tasks

### Task Understanding

- **TaskParser**: Automatic classification of user queries
- **Constraint Extraction**: Identifying requirements in the task

## Customization

### Model Size

Adjust the model size by modifying the initialization parameters:

```python
model = TaskSolvingLLM(
    vocab_size=50257,
    dim=1024,         # Increase for larger models
    num_layers=24,    # Increase for deeper models
    num_heads=16,     # Increase for more attention heads
    # ...
)
```

### Memory Configuration

```python
model = TaskSolvingLLM(
    # ...
    memory_capacity=20000,     # Increase for more memories
    memory_path="custom_memories.json",  # Custom storage path
)
```

### Generation Parameters

```python
result = model.solve_task(
    query="Write a short story about a robot",
    thinking_mode=True,      # Force thinking mode
    max_new_tokens=1000,     # Generate longer responses
    temperature=0.8,         # Higher for more creative outputs
)
```

## Examples

### Math Problem

```
You: Solve the quadratic equation: 2x² + 5x - 3 = 0

Processing...

Thinking process:
- Understanding: This is a quadratic equation in the form ax² + bx + c = 0 where a=2, b=5, and c=-3.
- Step 1: I'll use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
- Step 2: Substituting the values: x = (-5 ± √(5² - 4×2×(-3))) / (2×2)
- Step 3: Simplifying: x = (-5 ± √(25 + 24)) / 4 = (-5 ± √49) / 4 = (-5 ± 7) / 4
- Verification: Let's check our answers. For x = 0.5: 2(0.5)² + 5(0.5) - 3 = 2(0.25) + 2.5 - 3 = 0.5 + 2.5 - 3 = 0
For x = -3: 2(-3)² + 5(-3) - 3 = 2(9) - 15 - 3 = 18 - 15 - 3 = 0

Response: The solutions to the equation 2x² + 5x - 3 = 0 are x = 0.5 and x = -3.

Task identified as: math_problem
```

### Code Generation

```
You: Write a Python function to find the longest palindromic substring in a given string

Processing...

Thinking process:
- Requirements: I need to write a Python function that finds the longest palindromic substring in a given string.
- Design: I'll use the expand around center approach, which is efficient for this problem. For each character, we'll expand outward to find palindromes.
- Implementation: [Code showing the implementation of the function]
- Testing: Let's test with examples like "babad" and "cbbd"

Response: 
```python
def longest_palindromic_substring(s):
    """
    Find the longest palindromic substring in a given string.
    
    Args:
        s (str): Input string
        
    Returns:
        str: Longest palindromic substring
    """
    if not s:
        return ""
        
    start = 0  # Start index of longest palindrome
    max_len = 1  # Length of longest palindrome
    
    # Helper function to expand around center
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1
    
    # Check each position as potential center
    for i in range(len(s)):
        # Expand for odd-length palindromes
        left1, right1 = expand_around_center(i, i)
        length1 = right1 - left1 + 1
        
        # Expand for even-length palindromes
        left2, right2 = expand_around_center(i, i + 1)
        length2 = right2 - left2 + 1
        
        # Update if we found a longer palindrome
        if length1 > max_len:
            max_len = length1
            start = left1
        
        if length2 > max_len:
            max_len = length2
            start = left2
    
    return s[start:start + max_len]

# Test cases
test_cases = ["babad", "cbbd", "a", "ac", "racecar", "bananas"]
for test in test_cases:
    print(f"Input: '{test}', Longest palindrome: '{longest_palindromic_substring(test)}'")
```

This function has O(n²) time complexity and O(1) space complexity. It works by expanding around each possible center (both single characters and between characters) to find palindromes.

Task identified as: code_generation
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Inspired by recent advances in LLM architectures
- Implementation draws from transformer design principles
- Cognitive components based on research in reasoning in language models
