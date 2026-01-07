extern crate log;

use std::num::NonZeroU32;

use std::{io::Write, sync::Arc};

use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, LlamaContextLoadError,
    LlamaCppError, LlamaModelLoadError, LogOptions, NewLlamaChatMessageError, StringToTokenError,
    TokenToStringError,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::{BatchAddError, LlamaBatch},
    model::{
        AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special, params::LlamaModelParams,
    },
    sampling::LlamaSampler,
    send_logs_to_tracing,
};

use rust_dynamic::types::*;
use rust_dynamic::value::Value;

//
// Model context length (in tokens) used during inference.
//
pub const DEFAULT_CONTEXT_LENGTH: usize = 4096 * 4;

//
// Batch size used during inference.
//
pub const DEFAULT_BATCH_SIZE: usize = 4096 * 4;

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
    registry: DeepThoughtBackend,
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    messages: Vec<LlamaChatMessage>,
}

pub struct DeepThought {
    pub backend: DeepThoughtBackend,
    pub model: DeepThoughtModel,
}

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
        let chat_template = model.chat_template(None)?;

        Ok(DeepThoughtModel {
            registry: self.clone(),
            batch_size: DEFAULT_BATCH_SIZE,
            context_length: DEFAULT_CONTEXT_LENGTH,
            model,
            chat_template,
            messages: vec![LlamaChatMessage::new(
                "system".to_string(),
                system_prompt.to_string(),
            )?],
        })
    }
}

impl DeepThoughtModel {
    pub fn reset_messages(&mut self, system_prompt: &str) -> Result<(), Error> {
        self.messages.clear();
        self.messages.push(LlamaChatMessage::new(
            "system".to_string(),
            system_prompt.to_string(),
        )?);
        Ok(())
    }

    pub fn send_with_history(
        &mut self,
        prompt: &str,
        output: &mut impl Write,
    ) -> Result<(), Error> {
        let mut inference = vec![];
        self.infer(prompt, &mut inference, true)?;
        let inference = match String::from_utf8(inference) {
            Ok(inference) => inference,
            Err(err) => return Err(format!("{}", err).into()),
        };

        self.messages.push(LlamaChatMessage::new(
            "user".to_string(),
            prompt.to_string(),
        )?);

        self.messages.push(LlamaChatMessage::new(
            "assistant".to_string(),
            inference.to_string(),
        )?);

        output.write_all(inference.as_bytes())?;

        Ok(())
    }

    pub fn send_without_history(
        &mut self,
        prompt: &str,
        output: &mut impl Write,
    ) -> Result<(), Error> {
        let mut inference = vec![];
        self.infer(prompt, &mut inference, false)?;
        let inference = match String::from_utf8(inference) {
            Ok(inference) => inference,
            Err(err) => return Err(format!("{}", err).into()),
        };

        output.write_all(inference.as_bytes())?;

        Ok(())
    }

    fn infer(&mut self, prompt: &str, output: &mut impl Write, history: bool) -> Result<(), Error> {
        if history {
            self.messages.push(LlamaChatMessage::new(
                "user".to_string(),
                prompt.to_string(),
            )?);
        }

        let prompt = self
            .model
            .apply_chat_template(&self.chat_template, &self.messages, true)?;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;

        let context_params = LlamaContextParams::default()
            .with_n_batch(self.batch_size as u32)
            .with_n_ctx(NonZeroU32::new(self.context_length as u32));
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        let n_len = self.context_length as i32;
        let n_cxt = context.n_ctx() as i32;
        let n_kv_req = tokens.len() as i32 + (n_len - tokens.len() as i32);
        if n_kv_req > n_cxt {
            return Err(
                Error::InternalNativeError(
                    "n_kv_req > n_ctx, the required kv cache size is not big enough either reduce n_len or increase n_ctx".to_string()));
        }

        let mut batch = LlamaBatch::new(self.batch_size, 1);

        let last_index = tokens.len() - 1;
        for (i, token) in tokens.into_iter().enumerate() {
            batch.add(token, i as i32, &[0], i == last_index)?;
        }
        context.decode(&mut batch)?;

        // Decode and sample tokens.
        let mut n_cur = batch.n_tokens();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(1337),
        ]);
        while n_cur <= n_len {
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                eprintln!();
                break;
            }

            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            output.write_all(&output_bytes)?;
            output.flush()?;

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;

            match context.decode(&mut batch) {
                Ok(_) => {}
                Err(_) => return Err(Error::InternalNativeError("Decoding error".to_string())),
            };
        }

        let _ = self.messages.pop();

        Ok(())
    }

    pub fn chat(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        let mut output = vec![];
        match self.send_with_history(prompt, &mut output) {
            Ok(_) => {
                return Ok(String::from_utf8_lossy(&output).to_string());
            }
            Err(err) => easy_error::bail!("{:?}", err),
        }
    }

    pub fn ask(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        let mut output = vec![];
        match self.send_without_history(prompt, &mut output) {
            Ok(_) => {
                return Ok(String::from_utf8_lossy(&output).to_string());
            }
            Err(err) => easy_error::bail!("{:?}", err),
        }
    }
}

impl DeepThought {
    pub fn new(gguf_model: &str) -> Result<Self, easy_error::Error> {
        let backend = match DeepThoughtBackend::new() {
            Ok(backend) => backend,
            Err(err) => {
                easy_error::bail!("BACKEND ERROR: {:?}", err);
            }
        };
        let model = match backend.load_model(gguf_model, "You are the robot!") {
            Ok(model) => model,
            Err(err) => {
                easy_error::bail!("MODEL ERROR: {:?}", err);
            }
        };
        Ok(DeepThought {
            backend: backend,
            model: model,
        })
    }
    pub fn chat(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        self.model.chat(prompt)
    }

    pub fn ask(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        self.model.ask(prompt)
    }

    pub fn c(&mut self, prompt: Value) -> Result<Value, easy_error::Error> {
        let prompt_str = match prompt.conv(STRING) {
            Ok(str_val) => match str_val.cast_string() {
                Ok(prompt_str) => prompt_str,
                Err(err) => easy_error::bail!("Error in prompt casting: {}", err),
            },
            Err(err) => easy_error::bail!("Error in prompt conversion: {}", err),
        };
        match self.chat(&prompt_str) {
            Ok(res) => return Ok(Value::from_str(&res)),
            Err(err) => easy_error::bail!("LLAMA.CPP error: {}", err),
        };
    }

    pub fn a(&mut self, prompt: Value) -> Result<Value, easy_error::Error> {
        let prompt_str = match prompt.conv(STRING) {
            Ok(str_val) => match str_val.cast_string() {
                Ok(prompt_str) => prompt_str,
                Err(err) => easy_error::bail!("Error in prompt casting: {}", err),
            },
            Err(err) => easy_error::bail!("Error in prompt conversion: {}", err),
        };
        match self.ask(&prompt_str) {
            Ok(res) => return Ok(Value::from_str(&res)),
            Err(err) => easy_error::bail!("LLAMA.CPP error: {}", err),
        };
    }
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
