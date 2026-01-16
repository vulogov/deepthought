extern crate log;

use easy_error::bail;

use crate::*;

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are helpful assistant.";

impl DeepThoughtRouterBuilder {
    pub fn default() -> Self {
        DeepThoughtRouterBuilder {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            prompt_model: None,
            default_embed_model: None,
        }
    }

    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = system_prompt.to_string();
        self
    }

    pub fn default_embed_model(mut self, default_embed_model: &str) -> Self {
        self.default_embed_model = Some(default_embed_model.to_string());
        self
    }

    pub fn build(self) -> Result<DeepThoughtRouter, easy_error::Error> {
        let prompt_model = match self.prompt_model {
            Some(prompt_model) => prompt_model,
            None => bail!("Prompt model not set"),
        };

        let default_embed_model = match self.default_embed_model {
            Some(default_embed_model) => default_embed_model,
            None => bail!("Default embed model not set"),
        };

        let mut router = match DeepThoughtRouter::new() {
            Ok(router) => router,
            Err(err) => bail!("Failed to create router: {}", err),
        };
        router.prompt_model = match router
            .backend
            .load_context_model(&prompt_model, &self.system_prompt)
        {
            Ok(model) => Some(model),
            Err(err) => bail!("Failed to load prompt model: {:?}", err),
        };
        match router.embed_model(&default_embed_model) {
            Ok(model) => Some(model),
            Err(err) => bail!("Failed to load default embed model: {}", err),
        };
        Ok(router)
    }
}
