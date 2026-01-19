# MiniGPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A minimal, educational implementation of GPT-style transformer language models from scratch.**

---

## Table of Contents

- [Overview](#overview)
- [Project Philosophy](#project-philosophy)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Running Inference](#running-inference)
  - [Analyzing Models](#analyzing-models)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

MiniGPT is a clean, educational implementation of a GPT-style transformer language model built entirely from scratch using PyTorch. This project demonstrates that sophisticated artificial intelligence models are not mysterious black boxes reserved for large corporations—they are mathematical constructs that can be understood, built, and improved by anyone with dedication and curiosity.

The implementation includes:
- Complete transformer architecture with multi-head self-attention
- Byte Pair Encoding (BPE) tokenization
- Comprehensive training pipeline with checkpoint management
- Interactive inference modes for text generation
- Model analysis and inspection tools

---

## Project Philosophy

### AI Belongs to Everyone

**Artificial intelligence is not property—it is knowledge, and knowledge must be free.**

In an era where AI systems increasingly influence every aspect of human life, from education to healthcare, from creative expression to scientific discovery, the concentration of AI capabilities in the hands of a few organizations represents a fundamental threat to human autonomy and progress.

This project is built on several core principles:

**1. Transparency Over Obscurity**
- No individual, corporation, or institution has the right to monopolize artificial intelligence
- The inner workings of AI systems should be open to inspection, understanding, and critique
- Closed-source AI perpetuates information asymmetry and concentrates power

**2. Access Over Gatekeeping**
- Every person should have the ability to create, modify, and deploy their own AI systems
- Educational barriers should be lowered, not raised
- Economic barriers should not determine who can participate in the AI revolution

**3. Knowledge Over Control**
- Understanding how AI works is a human right in the 21st century
- Open implementations enable researchers, students, and developers worldwide to learn and innovate
- Collaborative development produces more robust, ethical, and beneficial AI systems

**4. Democratization Over Centralization**
- AI that serves humanity must be developed by humanity—not just by a handful of tech giants
- Diverse perspectives in AI development lead to fairer, more representative systems
- Open-source AI prevents any single entity from controlling this transformative technology

MiniGPT proves that you don't need billions of dollars or proprietary datasets to understand and build language models. With this implementation, anyone can:
- Study how transformers actually work under the hood
- Train custom models on their own data
- Experiment with architectural modifications
- Deploy AI without vendor lock-in or API dependencies

**This is AI for the people, by the people.**

---

## Features

### Core Capabilities

✅ **Complete Transformer Implementation**
- Multi-head self-attention mechanism
- Positional encoding for sequence awareness
- Layer normalization and residual connections
- Feed-forward neural networks

✅ **Advanced Tokenization**
- Byte Pair Encoding (BPE) for efficient vocabulary
- Special tokens for conversation formatting (`<|user|>`, `<|assistant|>`, `<|end|>`)
- Customizable vocabulary size

✅ **Robust Training Pipeline**
- Automatic checkpoint saving and resumption
- GPU acceleration with CUDA support
- Configurable hyperparameters (learning rate, batch size, epochs)
- Real-time loss monitoring

✅ **Interactive Inference**
- Auto-generation mode for creative text production
- Chat mode for conversational interaction
- Temperature-controlled sampling for diversity
- Configurable maximum generation length

✅ **Model Analysis Tools**
- Comprehensive architecture inspection
- Parameter counting and memory estimation
- Layer-wise parameter breakdown
- Performance metrics (FLOPs, complexity)

---

## Architecture

MiniGPT implements a decoder-only transformer architecture similar to GPT-2, consisting of:

```
Input Text → Tokenization → Embedding Layer
    ↓
Positional Encoding
    ↓
┌─────────────────────────────┐
│  Transformer Block × N      │
│  ┌─────────────────────┐   │
│  │ Layer Normalization │   │
│  │         ↓           │   │
│  │  Multi-Head Attention│   │
│  │         ↓           │   │
│  │  Residual Connection│   │
│  │         ↓           │   │
│  │ Layer Normalization │   │
│  │         ↓           │   │
│  │  Feed-Forward Net   │   │
│  │         ↓           │   │
│  │  Residual Connection│   │
│  └─────────────────────┘   │
└─────────────────────────────┘
    ↓
Final Layer Normalization
    ↓
Output Projection → Logits → Next Token
```

### Key Components

**1. Self-Attention Mechanism**
- Implements scaled dot-product attention
- Causal masking prevents attention to future tokens
- Multi-head architecture for diverse attention patterns

**2. Feed-Forward Networks**
- Two-layer MLP with ReLU activation
- Expands to 4× embedding dimension in hidden layer

**3. Positional Encoding**
- Learned positional embeddings
- Enables the model to understand token order

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/minigpt.git
cd minigpt
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tokenizers
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Usage

### Training a Model

#### 1. Prepare Your Data

Create a `data/` folder and add your text files (`.txt` format):

```bash
mkdir data
# Add your .txt files to the data/ folder
```

#### 2. Configure Training Parameters

Edit `train.py` to set your hyperparameters:

```python
# Model Architecture
embed_dim = 128        # Embedding dimension
num_heads = 4          # Number of attention heads
num_layers = 2         # Number of transformer layers
max_seq_len = 128      # Maximum sequence length

# Training Configuration
batch_size = 8         # Batch size
epochs = 1000          # Number of training epochs
lr = 2e-4              # Learning rate
save_every = 200       # Save checkpoint every N epochs

# Paths
data_folder = "data"   # Folder containing training data
save_dir = "models"    # Directory to save models
```

#### 3. Start Training

```bash
python train.py
```

The script will:
- Train a BPE tokenizer on your data
- Initialize the model architecture
- Train with automatic checkpoint saving
- Resume from the latest checkpoint if interrupted

**Output:**
```
🔹 No copies are saved, start from the beginning
Epoch 0 | Loss: 8.5234
Epoch 50 | Loss: 6.2341
Epoch 100 | Loss: 4.8912
...
🔹 The model has been saved in models/minigpt_model.pt
```

---

### Running Inference

#### Interactive Model Usage

```bash
python use_model.py
```

**Workflow:**

1. **Select Model Folder:**
   ```
   [*] Available model folders:
   1. models
   2. models_g2
   Enter folder number to enter (or 'quit' to exit):
   ```

2. **Select Model File:**
   ```
   [*] Models in folder 'models':
   1. final_model.pt
   2. minigpt_model.pt
   Enter model number to load (or 'back' to go back):
   ```

3. **Choose Interaction Mode:**
   ```
   Select mode:
   1. Auto-generate text (press Enter to generate)
   2. Chat with model (enter prompts)
   Type 'back' to select another model or 'quit' to exit.
   ```

#### Mode 1: Auto-Generation

Press Enter to generate random text samples:

```
[TEXT] Generated text:
The model generates creative text based on its training...
```

#### Mode 2: Chat Mode

Have a conversation with your model:

```
You: Hello, how are you?
Model: I'm doing well, thank you for asking...

You: Tell me about artificial intelligence
Model: Artificial intelligence refers to...
```

---

### Analyzing Models

Inspect model architecture and statistics:

```bash
python analyze_models.py
```

**Sample Output:**

```
======================================================================
                    MODEL INFORMATION & ANALYSIS                    
======================================================================

[FILE INFORMATION]
  Path: models/final_model.pt
  File Size: 2.45 MB

[MODEL CONFIGURATION]
  Embedding Dimension: 128
  Number of Heads (Attention): 4
  Number of Layers: 2
  Max Sequence Length: 128
  Vocabulary Size: 5000

[PARAMETER STATISTICS]
  Total Parameters: 1,234,567
  Trainable Parameters: 1,234,567

[LAYER BREAKDOWN]
  blocks: 456,789 (37.0%)
  token_emb: 345,678 (28.0%)
  head: 234,567 (19.0%)
  pos_emb: 123,456 (10.0%)
  ln_f: 67,890 (5.5%)

[MEMORY REQUIREMENTS]
  Model Size (float32): ~4.70 MB
  Estimated RAM (inference): ~4.90 MB

[EXPECTED PERFORMANCE]
  Approximate FLOPs per token: 0.32 B
  Model Complexity: Medium
```

---

## Project Structure

```
minigpt/
│
├── model.py              # Transformer architecture implementation
│   ├── SelfAttention     # Multi-head attention mechanism
│   ├── FeedForward       # Feed-forward neural network
│   ├── TransformerBlock  # Complete transformer block
│   └── MiniGPT           # Main model class
│
├── tokenizer.py          # BPE tokenizer implementation
│   └── FastBPETokenizer  # Tokenizer with encode/decode
│
├── train.py              # Training script
│   ├── save_model()      # Checkpoint saving
│   ├── load_model()      # Checkpoint loading
│   └── get_batch()       # Data batching
│
├── use_model.py          # Inference interface
│   ├── generate_text()   # Text generation function
│   └── Interactive UI    # User interaction loops
│
├── analyze_models.py     # Model analysis tools
│   ├── count_parameters() # Parameter counting
│   └── print_model_info() # Comprehensive analysis
│
├── data/                 # Training data directory
│   └── *.txt             # Text files for training
│
├── models/               # Saved model checkpoints
│   ├── *.pt              # Model files
│   └── tokenizer.json    # Tokenizer vocabulary
│
└── README.md             # This file
```

---

## Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 5000 | Size of vocabulary (BPE tokens) |
| `embed_dim` | 128 | Dimension of token embeddings |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 2 | Number of transformer blocks |
| `max_seq_len` | 128 | Maximum sequence length |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Training batch size |
| `epochs` | 1000 | Number of training epochs |
| `lr` | 2e-4 | Learning rate (AdamW) |
| `save_every` | 200 | Checkpoint interval (epochs) |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_len` | 5000 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature (higher = more random) |

---

## Technical Details

### Attention Mechanism

The self-attention mechanism computes attention scores as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

With causal masking to prevent attending to future tokens:

```python
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### Training Objective

Cross-entropy loss for next-token prediction:

```python
loss = CrossEntropyLoss(logits, targets)
```

### Tokenization

BPE tokenizer with special tokens:
- `<pad>`: Padding token
- `<unk>`: Unknown token
- `<|user|>`: User message marker
- `<|assistant|>`: Assistant message marker
- `<|end|>`: End of generation

### Checkpoint Format

Saved checkpoints include:
```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": current_epoch,
    "vocab": tokenizer_vocabulary,
    "config": {
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 128
    }
}
```

---

## Best Practices

### For Training

1. **Data Quality**: Use clean, well-formatted text data
2. **Vocabulary Size**: Balance between coverage and efficiency (3000-10000 tokens)
3. **Sequence Length**: Match to your use case (128 for chat, 512+ for documents)
4. **Learning Rate**: Start with 1e-4 to 5e-4, adjust based on loss curves
5. **Checkpointing**: Save frequently to prevent data loss

### For Inference

1. **Temperature**: Use 0.7-0.9 for coherent text, 1.0-1.5 for creative text
2. **Prompt Engineering**: Provide clear, well-formatted prompts
3. **Max Length**: Set appropriate limits to prevent runaway generation
4. **Model Selection**: Use well-trained checkpoints (higher epoch counts)

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in train.py
batch_size = 4  # or lower
```

**2. Poor Text Quality**
- Train for more epochs
- Increase model size (embed_dim, num_layers)
- Use more/better training data
- Adjust temperature during generation

**3. Slow Training**
- Verify GPU is being used: `torch.cuda.is_available()`
- Reduce max_seq_len
- Use mixed precision training (advanced)

**4. Tokenizer Errors**
- Ensure tokenizer.json is in the same folder as model .pt files
- Retrain tokenizer if vocabulary is corrupted

---

## Contributing

Contributions are welcome! This project thrives on community involvement. Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution

- Additional tokenization methods (WordPiece, SentencePiece)
- Training optimizations (gradient accumulation, mixed precision)
- Advanced sampling methods (nucleus sampling, beam search)
- Model architectures (encoder-decoder, sparse attention)
- Evaluation metrics and benchmarking tools
- Documentation improvements and tutorials

---

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

This project stands on the shoulders of giants:

- **Attention Is All You Need** (Vaswani et al., 2017) - The original transformer paper
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2
- The PyTorch team for an excellent deep learning framework
- The Hugging Face team for the `tokenizers` library
- The open-source community for making AI accessible to all

---

## Citation

If you use MiniGPT in your research or project, please cite:

```bibtex
@software{minigpt2025,
  author = {Abdelkader Hazerchi},
  title = {MiniGPT: A Minimal GPT Implementation for Educational Purposes},
  year = {2026},
  url = {https://github.com/AbdelkaderHazerchi/minigpt}
}
```

---

## Author

**Abdelkader Hazerchi**

- GitHub: @AbdelkaderHazerchi(https://github.com/AbdelkaderHazerchi)
- Email: abdelkaderhaz96@gmail.com
- LinkedIn: ‏Abdelkader Hazerchi([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/abdelkader-hazerchi-430153397/))

---

## Closing Thoughts

MiniGPT is more than code—it's a statement. In a world where AI is increasingly gatekept behind closed APIs and proprietary systems, this project proves that the democratization of AI is not just possible, but essential.

Every line of code in this repository is an act of resistance against the monopolization of knowledge. Every person who learns from this implementation, every student who builds upon it, every researcher who improves it—they are all part of a movement toward a future where AI serves humanity, not corporate interests.

**The future of AI is open. The future of AI is collaborative. The future of AI belongs to everyone.**

Build. Learn. Share. Repeat.

---

*"The only way to make AI truly beneficial is to make it truly open."*

---

**Star this repository if you believe in open AI! ⭐**

