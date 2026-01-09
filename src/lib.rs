extern crate log;

use std::sync::{Arc, RwLock};

use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, LlamaContextLoadError,
    LlamaCppError, LlamaModelLoadError, LogOptions, NewLlamaChatMessageError, StringToTokenError,
    TokenToStringError,
    llama_backend::LlamaBackend,
    llama_batch::BatchAddError,
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel},
};
use rust_dynamic::types::*;
use rust_dynamic::value::Value;
use vecstore::VecStore;

pub mod deepthought;
pub mod deepthought_backend;
pub mod deepthought_builder;
pub mod deepthought_model;
pub mod deepthought_vector;

type DeepThoughtVector = Arc<RwLock<VecStore>>;

//
// LLAMA.CPP model wrapper
//
#[derive(Clone)]
pub struct DeepThoughtBackend {
    backend: Arc<LlamaBackend>,
}

pub struct DeepThoughtModel {
    pub context_length: usize,
    pub batch_size: usize,
    pub registry: DeepThoughtBackend,
    pub model: LlamaModel,
    pub chat_template: Option<LlamaChatTemplate>,
    pub system_prompt: String,
    pub messages: Vec<LlamaChatMessage>,
}

pub struct DeepThought {
    pub dbpath: String,
    pub backend: DeepThoughtBackend,
    pub model: DeepThoughtModel,
    pub embed_model: Option<DeepThoughtModel>,
    pub embedding_doc_prefix: String,
    pub embedding_query_prefix: String,
    pub vecstore: Option<DeepThoughtVecStore>,
}

pub struct DeepThoughtBuilder {
    dbpath: Option<String>,
    context_length: Option<usize>,
    batch_size: Option<usize>,
    chat_model_gguf: Option<String>,
    embed_model_gguf: Option<String>,
    embedding_doc_prefix: String,
    embedding_query_prefix: String,
}

pub struct DeepThoughtVecStore {
    pub path: Option<String>,
    pub conn: DeepThoughtVector,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    TextTooLong(&'static str),

    /// An input was too large for the configured
    /// context size.
    ContextSize {
        maximum: usize,
        actual: usize,
    },

    /// An input was too large for the configured
    /// microbatch size.
    MicrobatchSize {
        maximum: usize,
        actual: usize,
    },

    InternalNativeError(String),
    IoError(String),
}

impl From<LlamaCppError> for Error {
    fn from(value: LlamaCppError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<LlamaModelLoadError> for Error {
    fn from(value: LlamaModelLoadError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<ChatTemplateError> for Error {
    fn from(value: ChatTemplateError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<ApplyChatTemplateError> for Error {
    fn from(value: ApplyChatTemplateError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<NewLlamaChatMessageError> for Error {
    fn from(value: NewLlamaChatMessageError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<LlamaContextLoadError> for Error {
    fn from(value: LlamaContextLoadError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<BatchAddError> for Error {
    fn from(value: BatchAddError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<StringToTokenError> for Error {
    fn from(value: StringToTokenError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<TokenToStringError> for Error {
    fn from(value: TokenToStringError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<DecodeError> for Error {
    fn from(value: DecodeError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<EmbeddingsError> for Error {
    fn from(value: EmbeddingsError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value.to_string())
    }
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}
