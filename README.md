# DeepThought: Rust LLM Inference Library

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
[![GitHub](https://img.shields.io/github/stars/vulogov/deepthought)](https://github.com/vulogov/deepthought)

**DeepThought** is a powerful Rust library that provides a high-level interface for Large Language Model (LLM) inference, built as a wrapper around the `llama_cpp_2` crate. Inspired by the supercomputer from *The Hitchhiker's Guide to the Galaxy*, it offers deep computational thinking capabilities with Rust's safety and performance.

## ✨ Features

- 🚀 **Efficient LLM Inference** - Wrapper for llama.cpp via `llama_cpp_2`
- 🔄 **Dynamic Data Handling** - Seamless integration with `rust_dynamic::Value`
- ⚙️ **Flexible Configuration** - Customizable context and batch sizes
- 📚 **Vector Store Support** - Built-in embeddings and RAG capabilities
- 🛣️ **Advanced Routing** - Router system for multi-model management
- 💬 **Chat & Q&A Modes** - Multiple inference interfaces
- 📝 **Prompt Engineering** - Template-based prompt refinement
- 🔒 **Error Handling** - Comprehensive error types for all operations

## 📦 Installation

Since the crate is not yet published on crates.io, you can install it directly from Git:

### Option 1: Git Dependency
Add to your `Cargo.toml`:
```toml
[dependencies]
deepthought = { git = "https://github.com/vulogov/deepthought.git" }
rust_dynamic = "0.48.0"
```

### Option 2: Local Build
```bash
git clone https://github.com/vulogov/deepthought.git
cd deepthought
cargo build --release
```

## 🚀 Quick Start

### Basic Inference
```rust
use deepthought::DeepThought;
use rust_dynamic::Value;

// Initialize with your GGUF model
let mut dt = DeepThought::new("/path/to/model.gguf").unwrap();

// Chat completion
let response = dt.chat("Hello, how are you?", None).unwrap();
println!("{}", response);

// Q&A style inference
let answer = dt.ask("What is Rust programming?").unwrap();

// Using dynamic values
let question = Value::from_string("What is 42?".to_string());
let answer = dt.c(question).unwrap();
```

### Vector Store (RAG) Operations
```rust
// Add documents to vector store
dt.add_document("document.txt", Some(&["technical", "rust"])).unwrap();
dt.add_string("Rust is a systems programming language", Some(&["definition"])).unwrap();

// Query similar content
let results = dt.query("programming language", 5).unwrap();
for result in results {
    println!("Found: {}", result);
}
```

## 🏗️ Project Structure

```
deepthought/
├── src/
│   ├── lib.rs                    # Main library entry point
│   ├── deepthought_backend.rs    # Backend management
│   ├── deepthought_builder.rs    # Builder pattern
│   ├── deepthought_context.rs    # Conversation context
│   ├── deepthought_ctx_model.rs  # Context-based models
│   ├── deepthought_model.rs      # Core model operations
│   ├── deepthought_prompt.rs     # Prompt recommendations
│   ├── deepthought_router.rs     # Request routing
│   ├── deepthought_vector.rs     # Vector store operations
│   └── ... (router submodules)
├── examples/                     # Usage examples
├── tests/                       # Test suite
└── Cargo.toml                   # Project dependencies
```

## 📚 Core Components

### 1. **DeepThought** (Main Struct)
The primary interface for all LLM operations.

**Key Methods:**
- `new(path: &str) -> Result<Self>` - Creates instance with model
- `chat(prompt: &str, system_prompt: Option<&str>) -> Result<String>` - Chat inference
- `ask(prompt: &str) -> Result<String>` - Q&A inference
- `embed(text: &str) -> Result<Vec<f32>>` - Text embeddings
- `add_document(path: &str, tags: Option<&[&str]>) -> Result<()>` - Add to vector store
- `query(query: &str, top_k: usize) -> Result<Vec<Value>>` - Query vector store

### 2. **DeepThoughtModel**
Represents a loaded LLM model with inference capabilities.

### 3. **DeepThoughtVector**
Thread-safe vector store implementation for RAG applications.

### 4. **DeepThoughtRouter**
Advanced routing system for multi-model and session management.

## ⚙️ Configuration

### Default Parameters
- **Context Size**: 16384 tokens
- **Batch Size**: 16384 tokens
- **Embedding Model**: Optional separate model for embeddings

### Using Builder Pattern
```rust
use deepthought::DeepThoughtBuilder;

let dt = DeepThoughtBuilder::new()
    .with_model_path("/path/to/model.gguf")
    .with_context_size(8192)
    .with_embedding_model("/path/to/embedding.gguf")
    .build()?;
```

## 🔧 Building and Testing

### Building
```bash
# Standard build
cargo build

# Release build
cargo build --release

# Using Makefile
make all
```

### Testing
```bash
# Run all tests
cargo test

# Test with specific model
DEEPTHOUGHT_TEST_MODEL=/path/to/model.gguf cargo test

# Makefile commands
make test      # Run tests
make clean     # Clean artifacts
make rebuild   # Clean and rebuild
```

## 📊 Recommended Test Models

| Model | Size | URL |
|-------|------|-----|
| Llama-3.2-3B-Instruct-Q6_K | 3B | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Athena-1-0.5B-Q4_K_M | 0.5B | [Download](https://huggingface.co/itlwas/Athena-1-0.5B-Q4_K_M-GGUF) |
| Qwen2.5-0.5B-Instruct-Q3_K_M | 0.5B | [Download](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) |

## 📋 Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `llama-cpp-2` | 0.1.131 | Core LLM inference |
| `rust_dynamic` | 0.48.0 | Dynamic value system |
| `log` | 0.4.29 | Logging framework |
| `easy-error` | 1.0.0 | Error handling |
| `lazy_static` | 1.5.0 | Static initialization |
| `serde` | ^1.0 | Serialization |
| `nanoid` | ^0.4 | ID generation |

## 📝 API Documentation

Generate full API documentation:
```bash
cargo doc --open
```

This will build and open comprehensive documentation for all public components including:
- Core structs and their methods
- Error types and handling
- Router system components
- Vector store operations

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines:**
- Follow Rust conventions and best practices
- Include tests for new functionality
- Update documentation as needed
- Use descriptive commit messages

## 📄 License

This project is released under the **Unlicense**, dedicating the work to the public domain. See the [LICENSE](LICENSE) file for details.

## 🐛 Issues and Support

Found a bug or need help?
1. Check existing [issues](https://github.com/vulogov/deepthought/issues)
2. Create a new issue with detailed description
3. Include reproduction steps and environment details

## 📈 Performance Tips

1. **Use Quantized Models**: Q4_K_M or Q6_K GGUF models offer good performance/accuracy balance
2. **Adjust Context Size**: Reduce from default 16384 if memory is constrained
3. **Batch Operations**: Use vector stores for batch document processing
4. **Embedding Models**: Use specialized embedding models for RAG applications

## 🔮 Roadmap

- [X] Publish to crates.io
- [ ] GPU acceleration support
- [ ] Additional vector store backends
- [ ] Extended model format support
- [ ] More examples and tutorials

## 🙏 Credits

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Curtana project**: Initial inspiration and code ideas
- **Vladimir Uogov (vulogov)**: Project maintainer

---

**DeepThought** - Making LLM inference in Rust as simple as asking the ultimate question about life, the universe, and everything.

*Note: This is an early-stage project (version 0.1.x). APIs may change as development continues.*
