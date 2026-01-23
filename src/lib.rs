extern crate log;

use rust_rule_engine::{Facts, KnowledgeBase};
use serde::Deserialize;
use std::collections::HashMap;
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
pub mod deepthought_context;
pub mod deepthought_ctx_model;
pub mod deepthought_model;
pub mod deepthought_prompt;
pub mod deepthought_router;
pub mod deepthought_router_builder;
pub mod deepthought_router_catalog;
pub mod deepthought_router_chat;
pub mod deepthought_router_expert;
pub mod deepthought_router_expert_facts;
pub mod deepthought_router_llm;
pub mod deepthought_router_prompt;
pub mod deepthought_router_route;
pub mod deepthought_router_sessions;
pub mod deepthought_router_template;
pub mod deepthought_vector;
pub mod deepthought_vector_output;

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

pub struct DeepThoughtCtxModel {
    pub context_length: usize,
    pub batch_size: usize,
    pub registry: DeepThoughtBackend,
    pub model: LlamaModel,
    pub chat_template: Option<LlamaChatTemplate>,
    pub system_prompt: String,
}

pub struct DeepThoughtContext {
    max_msg: Option<usize>,
    messages: Vec<LlamaChatMessage>,
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

pub struct DeepThoughtRouter {
    sessions: HashMap<String, DeepThoughtContext>,
    backend: DeepThoughtBackend,
    prompt_model: Option<DeepThoughtCtxModel>,
    embed_model: Option<DeepThoughtModel>,
    routes: HashMap<String, DeepThought>,
    ctx_routes: HashMap<String, DeepThoughtCtxModel>,
    catalog: Option<DeepThoughtVecStore>,
    query_preference: String,
    embedding_query_prefix: String,
    knowledge_base: Arc<KnowledgeBase>,
    facts: HashMap<String, Facts>,
}

#[derive(Clone)]
pub struct DeepThoughtRouterBuilder {
    system_prompt: String,
    prompt_model: Option<String>,
    catalog_path: Option<String>,
    embedding_query_prefix: String,
    default_embed_model: Option<String>,
    query_preference: Option<String>,
}

pub struct DeepThoughtBuilder {
    dbpath: Option<String>,
    context_length: Option<usize>,
    batch_size: Option<usize>,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    chat_model_gguf: Option<String>,
    embed_model_gguf: Option<String>,
    embedding_doc_prefix: String,
    embedding_query_prefix: String,
    alpha: f32,
    k: usize,
    max_score: f32,
}

pub struct DeepThoughtVecStore {
    pub path: Option<String>,
    pub conn: DeepThoughtVector,
    chunk_size: usize,
    chunk_overlap: usize,
    k: usize,
    alpha: f32,
    max_score: f32,
    embedding_prefix: String,
    templates: HashMap<String, String>,
}

#[derive(Deserialize, Debug)]
pub struct DeepThoughtRecommededPrompt {
    pub raw_prompt: String,
    pub clarifying_questions: Vec<String>,
    pub prompts: HashMap<String, String>,
    pub rationale_bullets: Vec<String>,
    pub suggested_parameters: HashMap<String, serde_json::Value>,
    pub quick_tests: Vec<String>,
}

pub struct VecStoreNeighbors {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
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
