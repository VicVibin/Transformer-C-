# Transformer LLM from Scratch

A complete implementation of a Transformer-based Large Language Model built from the ground up in C++, featuring GPU acceleration, custom automatic differentiation, and multi-head attention mechanisms.

## Features

- **Custom Backpropagation Engine**: Full computational graph with automatic differentiation
- **GPU Acceleration**: CUDA-powered matrix operations with multi-stream processing
- **Transformer Architecture**: Complete encoder-decoder implementation with:
  - Multi-head self-attention
  - Cross-attention mechanisms
  - Layer normalization
  - Feed-forward networks
  - Positional encoding
- **Advanced Optimizers**: Adam optimizer with momentum and bias correction
- **Text Processing**: Built-in tokenization and word2vec embeddings
- **Flexible Inference**: TopK sampling with configurable parameters

## Architecture Overview

The implementation follows the standard Transformer architecture:

```
Encoder: Input → Multi-Head Attention → Add & Norm → FFN → Add & Norm
Decoder: Input → Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm → Linear → Softmax
```

## Core Components

### Computational Graph
- **Node**: Base computation unit with forward/backward passes
- **Parameter**: Trainable weights with gradient tracking
- **AdamParameter**: Advanced optimizer with momentum and RMSprop
- **GraphOperations**: Complete set of differentiable operations

### Attention Mechanisms
- **AttentionHead**: Single attention head with Q, K, V projections
- **MultiHeadAttention**: Parallel multi-head processing with GPU streams
- **Cross-Attention**: Encoder-decoder attention for sequence-to-sequence tasks

## Quick Start

### Prerequisites
- CUDA-capable GPU
- CUDA toolkit
- C++17 compiler
- Custom dependencies: `MathGPU`, `Linear_Algebra`, `Machine_Learning` libraries

### Basic Usage

```cpp
#include "includes/backprop.h"

// Initialize components
GraphOperations go;
Linear_Algebra lin;
Math math;
Machine_Learning ml;

// Load and tokenize data
text words = Load("path/to/dataset");
text cleanedWords = read_words(words, 0, 50000);
Tokenization T = tokenizer(cleanedWords, embed_size);

// Create model components
MultiHeadAttention mha_enc(8, embed_size);      // 8 heads, embedding dimension
MultiHeadAttention mha_dec_self(8, embed_size);
MultiHeadAttention mha_dec_cross(8, embed_size);

// Define parameters
auto W1 = std::make_shared<AdamParameter>("W1", lin.RandMatrix(embed, hidden, uniform), 1e-3f);
auto b1 = std::make_shared<AdamParameter>("b1", lin.RandMatrix(1, hidden, uniform), 1e-3f);
// ... additional parameters

// Build_encoder, build decoder are function calls using the go class., Z1 = go.matmul(W1, input), go.add(Z1, ..), the forward function can be defined to your desired architecture
// Training loop
for(int epoch = 0; epoch < num_epochs; epoch++) {
    // Forward pass
    auto encoder_out = build_encoder(input, mha_enc, go, W1, b1, W2, b2);
    auto decoder_out = build_decoder(target, encoder_out, mha_dec_self, mha_dec_cross, go, ...);
    auto loss = go.SoftmaxCrossEntropy(decoder_out, target_labels);
    
    // Backward pass
    go.backprop(loss);
    
    // Update parameters
    for(auto& param : parameters) {
        param->update();
    }
    
    // Clear computational graph
    go.clear_graph(loss);
}
```

### Configuration Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `embed` | Embedding dimension (d_model) | 128-1024 |
| `num_heads` | Number of attention heads | 4-16 |
| `context_size` | Input sequence length | 32-512 |
| `hidden_size` | FFN hidden dimension | 256-4096 |
| `learning_rate` | Adam learning rate | 1e-4 to 1e-2 |
| `topk` | Inference sampling parameter | 1-10 |

## Training Process

1. **Data Preparation**: Load text corpus and create tokenization mapping
2. **Model Initialization**: Create multi-head attention layers and FFN parameters
3. **Training Loop**:
   - Sample random sequences from dataset
   - Apply positional encoding
   - Forward pass through encoder-decoder
   - Compute cross-entropy loss
   - Backpropagation through computational graph
   - Parameter updates using Adam optimizer
4. **Inference**: Generate text using TopK sampling

## Key Operations

### Matrix Operations
- `matmul()`: GPU-accelerated matrix multiplication with stream parallelization
- `add()`: Element-wise addition with broadcasting support
- `LayerNorm()`: Layer normalization with custom backward pass
- `transpose()`: Matrix transposition
- `concatenate()`: Multi-tensor concatenation

### Activation Functions
- `RelU()`: Rectified Linear Unit
- `Softmax()`: Softmax activation
- `SoftmaxCrossEntropy()`: Combined softmax and cross-entropy loss

### Attention Operations
- `compute()`: Self-attention computation
- `compute_d()`: Cross-attention (decoder) computation
- Stream-parallel execution across multiple attention heads

## Performance Optimizations

- **GPU Streams**: Parallel attention head computation
- **Memory Management**: Efficient graph clearing and memory reuse
- **Asynchronous Execution**: Multi-threaded attention head processing
- **Custom CUDA Kernels**: Optimized matrix operations

## Example Output

```
Epoch: 1000
Time taking for forward and backward pass: 0.234s
Epoch 0 Loss: 2.45
Beginning the inference
the quick brown fox jumps over the lazy dog and runs
_____________________
Time elapsed for inference 8 times: 0.089s
```

## Memory Requirements

- **Model Parameters**: ~10-100MB depending on configuration
- **GPU Memory**: 2-8GB recommended for training
- **Dataset**: Flexible, processes text files of any size

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce batch size or sequence length
- **NaN Values**: Lower learning rate or add gradient clipping
- **Slow Training**: Ensure GPU streams are properly utilized

### Debug Features
- Parameter gradient monitoring
- Loss tracking and validation
- Memory usage reporting
- Stream synchronization verification

## Contributing

This implementation serves as an educational foundation for understanding Transformer architectures. Key areas for extension:
- Gradient clipping and regularization
- Batch gradient descent
- Learning rate scheduling
- Beam search decoding
- Model checkpointing
- Multi-GPU training
- Efficient data transfer and management
- KV caching for decoding

## License
 All usage and modifications should be made open source

---

**Note**: This implementation prioritizes educational clarity over production efficiency. For deployment scenarios, I would suggest pointers instead of std::vectors for non dynamic matrices and vectors
