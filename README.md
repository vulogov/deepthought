# DeepThought

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## Description

DeepThought is a Rust library that serves as a wrapper around the `llama_cpp_2` crate, providing an interface for running large language model (LLM) inference using the llama.cpp framework. It integrates with `rust_dynamic` for handling dynamic data types, making it suitable for applications requiring flexible value manipulation during AI operations.

The name "DeepThought" is inspired by the supercomputer from *The Hitchhiker's Guide to the Galaxy*, symbolizing deep computational thinking and AI capabilities.

This project is in its early stages (version 0.1) and aims to simplify LLM integration in Rust projects.

## Features

- Wrapper for llama.cpp via `llama_cpp_2` for efficient LLM inference.
- Support for dynamic values using `rust_dynamic::Value`.
- Configurable context and batch sizes (defaults: 16384 tokens).
- Error handling for common llama.cpp operations like model loading, tokenization, and sampling.
- Logging integration with the `log` crate.
- Support for embeddings and vector stores.
- Router for managing multiple models and sessions.

## Installation

Since the crate is not yet published on crates.io, you can add it to your project by cloning the repository and building it locally.

1. Clone the repository:
   ```
   git clone https://github.com/vulogov/deepthought.git
   ```

2. Add it as a dependency in your `Cargo.toml`:
   ```
   [dependencies]
   deepthought = { path = "/path/to/deepthought" }
   ```

Alternatively, use it directly via Git in `Cargo.toml`:
   ```
   [dependencies]
   deepthought = { git = "https://github.com/vulogov/deepthought.git" }
   ```

Build your project with `cargo build`.

## Usage

DeepThought provides a `DeepThoughtBackend` struct for managing the LLM backend. Here's a basic example:

```rust
use deepthought::DeepThought;
use rust_dynamic::Value;

// Initialize the backend
let mut deep_thought = DeepThought::new("/path/to/gguf/model").unwrap();

// Example: Using dynamic values
let value = Value::from_string("What is the answer to life, the universe, and everything?".to_string());

let output = deep_thought.c(value).unwrap();
println!("{}", output);
```

For detailed API usage, refer to the source code in `src/lib.rs` or generate Rustdoc with `cargo doc --open`. Examples will be added as the project develops.

### Dependencies

- `easy-error`: 1.0.0
- `lazy_static`: 1.5.0
- `llama-cpp-2`: 0.1.131
- `log`: 0.4.29
- `rust_dynamic`: 0.48.0
- Additional: `serde`, `nanoid`, etc., as per Cargo.toml.

## Building and Testing

Use the provided Makefile for common tasks:

- `make all`: Build the library.
- `make test`: Run tests.
- `make clean`: Clean the build artifacts.
- `make rebuild`: Clean and rebuild.

Alternatively, use Cargo directly:
- `cargo build`
- `cargo test`
- `cargo clean`

## API Documentation

This section provides basic documentation for all public components of the crate, including modules, structs, enums, and key functions. For full details, refer to the Rustdoc generated from the source code.

### Modules

- **`deepthought`**: Core module containing the `DeepThought` struct for managing LLM inference, embeddings, and vector stores.
- **`deepthought_backend`**: Manages the underlying llama.cpp backend, including model loading.
- **`deepthought_builder`**: Builder pattern for configuring and creating `DeepThought` instances.
- **`deepthought_context`**: Handles conversation contexts for models.
- **`deepthought_ctx_model`**: Manages context-based model interactions.
- **`deepthought_model`**: Core model loading and inference operations.
- **`deepthought_prompt`**: Provides prompt recommendations and refinements.
- **`deepthought_router`**: Routes requests to different models or sessions.
- **`deepthought_router_builder`**: Builder for router configurations.
- **`deepthought_router_catalog`**: Manages catalogs of models and routes in the router.
- **`deepthought_router_chat`**: Handles chat functionalities within the router.
- **`deepthought_router_llm`**: LLM-specific operations in the router.
- **`deepthought_router_prompt`**: Prompt refinement in the router.
- **`deepthought_router_route`**: Manages individual routes in the router.
- **`deepthought_router_sessions`**: Session management for the router.
- **`deepthought_router_template`**: Template rendering for router responses.
- **`deepthought_vector`**: Operations for vector stores and embeddings.
- **`deepthought_vector_output`**: Formats vector output.

### Key Structs

- **`DeepThought`**: The main struct for LLM operations.
  - Fields: `dbpath` (database path), `backend` (LLM backend), `model` (primary model), `embed_model` (optional embedding model), `embedding_doc_prefix` (prefix for document embeddings), `embedding_query_prefix` (prefix for query embeddings), `vecstore` (optional vector store).
  - Methods: `new` (creates instance), `embed_model` (loads embedding model), `chat` (performs chat inference), `ask` (performs Q&A inference), `embed` (generates embeddings), `c` (chat with dynamic Value), `a` (ask with dynamic Value), `add_document` (adds doc to vector store), `add_string` (adds string to vector store), `add_value` (adds Value to vector store), `delete_value` (deletes from vector store), `query` (queries vector store), `query_templated` (templated query), `len` (vector store length), `register_template` (registers template), `sync` (syncs vector store).

- **`DeepThoughtBackend`**: Manages the llama.cpp backend.
  - Fields: `backend` (Arc-wrapped LlamaBackend).
  - Methods: `new` (creates backend), `load_model` (loads model), `load_context_model` (loads context model), `supports_mlock` (checks mlock support).

- **`DeepThoughtModel`**: Represents a loaded LLM model.
  - Fields: `registry` (backend reference), `batch_size` (batch size), `context_length` (context length), `model` (LlamaModel), `chat_template` (optional chat template), `system_prompt` (system prompt), `messages` (chat messages).
  - Methods: `chat` (chat inference), `ask` (Q&A inference), `embed` (embeddings).

- **`DeepThoughtCtxModel`**: Context-based model.
  - Fields: Similar to `DeepThoughtModel`, plus context-specific params.
  - Methods: Similar to `DeepThoughtModel`, with context handling.

- **`DeepThoughtRouter`**: Routes LLM requests.
  - Fields: Router-specific configurations.
  - Methods: Routing, session management, etc.

- **`DeepThoughtVector`**: Type alias for thread-safe vector store (Arc<RwLock<VecStore>>).

### Enums and Other Types

- Various error types from llama_cpp_2 integrations, such as `LlamaCppError`, `EmbeddingsError`, etc.
- Uses `rust_dynamic::Value` for dynamic data handling.

For comprehensive details, including parameters, return types, and errors for each method, run `cargo doc` on the crate.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. Ensure your code follows Rust best practices and includes tests where applicable. Add or update Rustdoc comments for new components.

## License

This project is released under the Unlicense, which dedicates the work to the public domain. See the [LICENSE](LICENSE) file for details.

## Contact

- Owner: Vladimir Uogov (vulogov on GitHub)

## Models

You can use the following models for crate testing:

- `Llama-3.2-3B-Instruct-Q6_K`: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf
- `Athena-1-0.5B-Q4_K_M`: https://huggingface.co/itlwas/Athena-1-0.5B-Q4_K_M-GGUF/resolve/main/athena-1-0.5b-q4_k_m.gguf?download=true
- `Qwen2.5-0.5B-Instruct-Q3_K_M`: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q3_k_m.gguf?download=true

## Credits

- `llama-cpp`: https://github.com/ggerganov/llama.cpp
- Initial ideas and code from `Curtana project`: https://lib.rs/crates/curtana
