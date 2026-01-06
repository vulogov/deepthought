extern crate log;

use std::num::NonZeroU32;

use std::{
    io::{Write},
    sync::Arc,
};

use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, LlamaCppError,
    LlamaContextLoadError, LlamaModelLoadError, NewLlamaChatMessageError,
    StringToTokenError, TokenToStringError,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::{BatchAddError, LlamaBatch},
    model::{
        AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special, params::LlamaModelParams,
    },
    sampling::LlamaSampler,
    send_logs_to_tracing,
    LogOptions,
};

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

pub struct DeepThoughModel {
    registry: DeepThoughtBackend,
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    messages: Vec<LlamaChatMessage>,
}

impl DeepThoughtBackend {
    pub fn new() -> Result<Self, Error> {
        lazy_static::lazy_static! {
            static ref LLAMA_BACKEND: Arc<LlamaBackend> = {
                send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
                Arc::new(LlamaBackend::init().unwrap())
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
   ) -> Result<DeepThoughModel, Error> {
       let model_params = LlamaModelParams::default();
       let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)?;
       let chat_template = model.chat_template(None)?;

       Ok(DeepThoughModel {
           registry: self.clone(),
           model,
           chat_template,
           messages: vec![LlamaChatMessage::new(
               "system".to_string(),
               system_prompt.to_string(),
           )?],
       })
   }
}


impl DeepThoughModel {
    pub fn send_with_history(
        &mut self,
        prompt: &str,
        output: &mut impl Write,
    ) -> Result<(), Error> {
        let mut inference = vec![];
        self.infer(prompt, &mut inference)?;
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

    pub fn infer(&mut self, prompt: &str, output: &mut impl Write) -> Result<(), Error> {

        self.messages.push(LlamaChatMessage::new(
            "user".to_string(),
            prompt.to_string(),
        )?);

        // Format and tokenize the prompt.
        let prompt = self
            .model
            .apply_chat_template(&self.chat_template, &self.messages, true)?;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;

        // Prepare inference context.
        let context_params = LlamaContextParams::default()
            .with_n_batch(DEFAULT_CONTEXT_LENGTH as u32)
            .with_n_ctx(NonZeroU32::new(DEFAULT_CONTEXT_LENGTH as u32));
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        // Make sure the KV cache is big enough to hold all the prompt and generated tokens.
        let n_len = DEFAULT_CONTEXT_LENGTH as i32;
        let n_cxt = context.n_ctx() as i32;
        let n_kv_req = tokens.len() as i32 + (n_len - tokens.len() as i32);
        if n_kv_req > n_cxt {
            return Err(
                Error::InternalNativeError(
                    "n_kv_req > n_ctx, the required kv cache size is not big enough either reduce n_len or increase n_ctx".to_string()));
        }

        // Group tokens into batches.
        let mut batch = LlamaBatch::new(DEFAULT_BATCH_SIZE, 1);

        // Submit initial batch for inference.
        let last_index = tokens.len() - 1;
        for (i, token) in tokens.into_iter().enumerate() {
            // llama_decode will output logits only for the last token of the prompt
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
            // sample the next token
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            sampler.accept(token);

            // is it an end of stream?
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

            context.decode(&mut batch).expect("failed to eval");
        }

        // Remove the prompt from the inference history.
        let _ = self.messages.pop();

        Ok(())
    }

    pub fn chat(
        &mut self,
        prompt: &str
    ) -> Result<String, easy_error::Error> {
        let mut output = vec![];
        match self.send_with_history(prompt, &mut output) {
            Ok(_) => {
                return Ok(String::from_utf8_lossy(&output).to_string());
            }
            Err(err) => easy_error::bail!("{:?}", err),
        }
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
