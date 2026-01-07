# DeepThought

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## Description

DeepThought is a Rust library that serves as a wrapper around the `llama_cpp_2` crate, providing an interface for running large language model (LLM) inference using the llama.cpp framework. It integrates with `rust_dynamic` for handling dynamic data types, making it suitable for applications requiring flexible value manipulation during AI operations.

The name "DeepThought" is inspired by the supercomputer from *The Hitchhiker's Guide to the Galaxy*, symbolizing deep computational thinking and AI capabilities.

This project is in its early stages (version 0.1.0) and aims to simplify LLM integration in Rust projects.

## Features

- Wrapper for llama.cpp via `llama_cpp_2` for efficient LLM inference.
- Support for dynamic values using `rust_dynamic::Value`.
- Configurable context and batch sizes (defaults: 16384 tokens).
- Error handling for common llama.cpp operations like model loading, tokenization, and sampling.
- Logging integration with the `log` crate.

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

DeepThought provides a `DeepThoughtBackend` struct for managing the LLM backend. Here's a basic example (based on the library's structure; adjust as needed):

```rust
use deepthought::DeepThought;
use rust_dynamic::Value;

// Initialize the backend (example; refer to lib.rs for full API)
let deep_thou    = DeepThought::new("/path/to/gguf/model");

// Example: Using dynamic values
let value = Value::from_string("What is the answer on life universe and everything".to_string());

let output = deep_thought.c(value);
```

For detailed API usage, refer to the source code in `src/lib.rs`. Examples will be added as the project develops.

### Dependencies

- `easy-error`: 1.0.0
- `lazy_static`: 1.5.0
- `llama-cpp-2`: 0.1.131
- `log`: 0.4.29
- `rust_dynamic`: 0.48.0

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

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. Ensure your code follows Rust best practices and includes tests where applicable.

## License

This project is released under the Unlicense, which dedicates the work to the public domain. See the [LICENSE](LICENSE) file for details.

## Contact

- Owner: Vladimir Uogov (vulogov on GitHub)

## Credit

- `llama-cpp`: https://github.com/ggerganov/llama.cpp
- Good portion of ideas and initial code came from `Curtana project`: https://lib.rs/crates/curtana
