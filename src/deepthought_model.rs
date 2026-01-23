extern crate log;

use crate::*;

use std::num::{NonZero, NonZeroU32};

// use easy_error::bail;
use std::io::Write;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, Special},
    sampling::LlamaSampler,
};

impl DeepThoughtModel {
    pub fn reset_messages(&mut self, system_prompt: Option<&str>) -> Result<(), Error> {
        self.messages.clear();
        let system_prompt = match system_prompt {
            Some(system_prompt) => system_prompt.to_string(),
            None => self.system_prompt.clone(),
        };
        self.messages.push(LlamaChatMessage::new(
            "system".to_string(),
            system_prompt.to_string(),
        )?);
        Ok(())
    }

    pub fn add_inference_to_prompt(&mut self, data: &str) -> Result<(), Error> {
        self.messages.push(LlamaChatMessage::new(
            "assistant".to_string(),
            data.to_string(),
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
        let chat_template = match self.chat_template {
            Some(ref template) => template.clone(),
            None => match LlamaChatTemplate::new("chatml") {
                Ok(template) => template,
                Err(err) => return Err(format!("{}", err).into()),
            },
        };
        if history {
            self.messages.push(LlamaChatMessage::new(
                "user".to_string(),
                prompt.to_string(),
            )?);
        }

        let prompt = self
            .model
            .apply_chat_template(&chat_template, &self.messages, true)?;
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

    pub fn embed(&self, text: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>, Error> {
        // Tokenize the text.
        let mut tokens = Vec::with_capacity(text.len());
        for text in text {
            tokens.push(self.model.str_to_token(text.as_ref(), AddBos::Always)?);
        }

        // Prepare inference context.
        let thread_count = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(1).unwrap())
            .get() as i32;
        let context_params = LlamaContextParams::default()
            .with_n_batch(self.context_length as u32)
            .with_n_ubatch(self.context_length as u32)
            .with_n_ctx(NonZeroU32::new(self.context_length as u32))
            .with_n_threads(thread_count)
            .with_n_threads_batch(thread_count)
            .with_embeddings(true);
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        // Make sure the KV cache is big enough to hold all the text.
        let n_ctx = context.n_ctx() as usize;
        let n_ubatch = context.n_ubatch() as usize;
        for tokens in &tokens {
            if n_ctx < tokens.len() {
                return Err(Error::ContextSize {
                    maximum: n_ubatch,
                    actual: tokens.len(),
                });
            } else if n_ubatch < tokens.len() {
                return Err(Error::MicrobatchSize {
                    maximum: n_ubatch,
                    actual: tokens.len(),
                });
            }
        }

        // Prepare a reusable batch.
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // TODO: @caer: include multiple tokens per batch if possible.
        // Embed batches.
        let mut embeddings = Vec::with_capacity(tokens.len());
        for tokens in tokens {
            batch.add_sequence(&tokens, 0, false)?;

            // Run inference for embedding.
            context.clear_kv_cache();
            context.decode(&mut batch)?;

            // Extract embedding from the model.
            let embedding = context.embeddings_seq_ith(0)?;

            // Normalize embedding.
            let embedding_magnitude = embedding
                .iter()
                .fold(0.0, |acc, &val| val.mul_add(val, acc))
                .sqrt();
            let embedding: Vec<_> = embedding
                .iter()
                .map(|&val| val / embedding_magnitude)
                .collect();

            embeddings.push(embedding);
        }
        Ok(embeddings)
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
