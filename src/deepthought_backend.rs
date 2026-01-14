extern crate log;

use crate::*;

use std::sync::Arc;

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_supports_mlock,
    model::{LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    send_logs_to_tracing,
};

//
// Model context length (in tokens) used during inference.
//
pub const DEFAULT_CONTEXT_LENGTH: usize = 4096 * 4;

//
// Batch size used during inference.
//
pub const DEFAULT_BATCH_SIZE: usize = 4096 * 4;

impl DeepThoughtBackend {
    pub fn new() -> Result<Self, Error> {
        lazy_static::lazy_static! {
            static ref LLAMA_BACKEND: Arc<LlamaBackend> = {
                send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
                let lb = match LlamaBackend::init() {
                    Ok(lb) => lb,
                    Err(err) => {
                        let msg = format!("Error creating LLAMA backend: {}", err);
                        log::error!("{}", &msg);
                        panic!("{}", &msg);
                    }
                };
                Arc::new(lb)
            };
        }

        Ok(Self {
            backend: LLAMA_BACKEND.clone(),
        })
    }

    pub fn load_model(
        &self,
        model_path: &str,
        system_prompt: &str,
    ) -> Result<DeepThoughtModel, Error> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)?;
        let chat_template = match model.chat_template(None) {
            Ok(template) => Some(template),
            Err(_) => None,
        };

        Ok(DeepThoughtModel {
            registry: self.clone(),
            batch_size: DEFAULT_BATCH_SIZE,
            context_length: DEFAULT_CONTEXT_LENGTH,
            model,
            chat_template,
            system_prompt: system_prompt.to_string(),
            messages: vec![LlamaChatMessage::new(
                "system".to_string(),
                system_prompt.to_string(),
            )?],
        })
    }

    pub fn load_context_model(
        &self,
        model_path: &str,
        system_prompt: &str,
    ) -> Result<DeepThoughtCtxModel, Error> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)?;
        let chat_template = match model.chat_template(None) {
            Ok(template) => Some(template),
            Err(_) => None,
        };

        Ok(DeepThoughtCtxModel {
            registry: self.clone(),
            batch_size: DEFAULT_BATCH_SIZE,
            context_length: DEFAULT_CONTEXT_LENGTH,
            model,
            chat_template,
            system_prompt: system_prompt.to_string(),
        })
    }

    pub fn supports_mlock() -> bool {
        llama_supports_mlock()
    }
}
