# GPT Language Model - Shakespeare Text Generator

A PyTorch implementation of a transformer-based language model for character-level text generation, trained on Shakespeare's complete works.

## Overview

This project implements a GPT-style transformer architecture with multi-head self-attention, enabling the model to learn long-range dependencies and generate coherent Shakespeare-like text. The model uses a causal attention mechanism to ensure each position can only attend to previous tokens, maintaining the autoregressive property needed for text generation.

## Features

✨ **Multi-head Self-Attention**: 6 parallel attention heads capturing different semantic relationships  
🔄 **Transformer Blocks**: Stacked 6-layer architecture with residual connections and layer normalization  
📊 **Efficient Training**: Mixed precision support ready, optimized for modern GPUs  
🎲 **Character-Level Generation**: Learns the distribution of Shakespeare's vocabulary and writing style  
⚡ **GPU Optimized**: CUDA-enabled training with proper device management  

## Architecture

### Model Components

```
GPTLanguageModel
├── Token Embedding (vocab_size → 384)
├── Position Embedding (block_size → 384)
├── 6x Transformer Block
│   ├── Multi-Head Attention (6 heads, 384 dim)
│   ├── Feed Forward Network (384 → 1536 → 384)
│   ├── Layer Normalization (x2)
│   └── Residual Connections
├── Final Layer Norm
└── Output Head (384 → vocab_size)
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Sequences processed in parallel |
| `block_size` | 256 | Maximum context length for predictions |
| `n_embd` | 384 | Embedding/hidden dimension |
| `n_head` | 6 | Number of attention heads |
| `n_layer` | 6 | Number of transformer blocks |
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `dropout` | 0.2 | Regularization parameter |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

## Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio

# Clone and navigate to repository
cd ./AI_security
```

## Usage

### Training

Run the training loop:

```bash
python gpt.py
```

The script will:
1. Load Shakespeare text from `input.txt`
2. Create character-level tokenization
3. Train for 5000 iterations with periodic evaluation
4. Display train/validation losses every 500 iterations

### Generating Text

To generate Shakespeare-style text, add this to the script after model training:

```python
# Start with a seed character
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate 500 new characters
generated = model.generate(context, max_new_tokens=500)

# Decode and print
print(decode(generated[0].tolist()))
```

## Dataset

The model trains on `input.txt` containing Shakespeare's complete works (~5MB of text). The dataset is automatically split:
- **90%** for training
- **10%** for validation

### Character Vocabulary

The model learns all unique characters appearing in Shakespeare's text, typically ~65 characters including:
- Lowercase & uppercase letters
- Digits
- Punctuation marks
- Whitespace

## Training Performance

- **Training Time**: ~1-2 hours on RTX 4070 (varies by hardware)
- **Total Parameters**: ~1.5M
- **Convergence**: Loss typically decreases from ~4.5 to ~1.5-2.0
- **GPU Memory**: ~4-5GB with current configuration

## Hardware Recommendations

| GPU | VRAM | Recommended |
|-----|------|-------------|
| RTX 3050 | 4GB | ⚠️ Tight fit |
| RTX 4060 | 8GB | ✅ Comfortable |
| RTX 4070 | 12GB | ✅ Ideal |
| Quadro P400 | 16GB | ✅ Good |
| A100 | 40GB | ✅ Overkill |

For CPU-only training, expect 10-50x slower execution.

## Model Capabilities

After training, the model can:
- ✅ Generate grammatically plausible sentences
- ✅ Maintain character consistency in dialogue
- ✅ Learn punctuation and formatting patterns
- ✅ Develop pseudo-Shakespeare vocabulary

## Technical Details

### Causal Masking

Attention weights are masked to prevent the model from attending to future tokens:
```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

This ensures the model learns to generate tokens sequentially without cheating by looking ahead.

### Position Encoding

Learned positional embeddings (not sinusoidal) help the model understand token positions up to `block_size=256`.

### Weight Initialization

Weights are initialized from normal distributions:
- Linear layers: N(0, 0.02)
- Embeddings: N(0, 0.02)
- Biases: zeros

This initialization helps with training stability.

## Files

- `gpt.py` - Main model implementation and training loop
- `input.txt` - Shakespeare training dataset
- `more.txt` - Example generated output

## Future Improvements

- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Add validation metrics (perplexity, BLEU scores)
- [ ] Distributed training across multiple GPUs
- [ ] Model checkpointing and resumable training
- [ ] Inference optimization (KV caching)
- [ ] Fine-tuning on custom datasets

## License

This project is provided as-is for educational and research purposes.
