extern crate log;

use crate::*;

use std::num::NonZeroU32;

use std::io::Write;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatTemplate, Special},
    sampling::LlamaSampler,
};

impl DeepThoughtCtxModel {
    pub fn send_with_history(
        &mut self,
        prompt: &str,
        ctx: &mut DeepThoughtContext,
        output: &mut impl Write,
    ) -> Result<(), Error> {
        let mut inference = vec![];
        self.infer(prompt, ctx, &mut inference)?;
        let inference = match String::from_utf8(inference) {
            Ok(inference) => inference,
            Err(err) => return Err(format!("{}", err).into()),
        };

        match ctx.user(prompt) {
            Ok(_) => {}
            Err(err) => return Err(format!("{}", err).into()),
        }
        match ctx.assistant(&inference.to_string()) {
            Ok(_) => {}
            Err(err) => return Err(format!("{}", err).into()),
        }

        output.write_all(inference.as_bytes())?;

        Ok(())
    }

    fn infer(
        &mut self,
        prompt: &str,
        ctx: &mut DeepThoughtContext,
        output: &mut impl Write,
    ) -> Result<(), Error> {
        let chat_template = match self.chat_template {
            Some(ref template) => template.clone(),
            None => match LlamaChatTemplate::new("chatml") {
                Ok(template) => template,
                Err(err) => return Err(format!("{}", err).into()),
            },
        };
        match ctx.user(prompt) {
            Ok(_) => {}
            Err(err) => return Err(format!("{}", err).into()),
        }
        let prompt = self
            .model
            .apply_chat_template(&chat_template, ctx.messages(), true)?;
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
        ctx.remove_last();
        Ok(())
    }

    pub fn chat(
        &mut self,
        prompt: &str,
        ctx: &mut DeepThoughtContext,
    ) -> Result<String, easy_error::Error> {
        let mut output = vec![];
        match self.send_with_history(prompt, ctx, &mut output) {
            Ok(_) => {
                return Ok(String::from_utf8_lossy(&output).to_string());
            }
            Err(err) => easy_error::bail!("{:?}", err),
        }
    }
}
