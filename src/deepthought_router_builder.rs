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
            embedding_query_prefix: None,
            query_preference: None,
            catalog_path: None,
        }
    }

    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = system_prompt.to_string();
        self
    }

    pub fn prompt_model(mut self, prompt_model: &str) -> Self {
        self.prompt_model = Some(prompt_model.to_string());
        self
    }

    pub fn default_embed_model(mut self, default_embed_model: &str) -> Self {
        self.default_embed_model = Some(default_embed_model.to_string());
        self
    }

    pub fn embedding_query_prefix(mut self, embedding_query_prefix: &str) -> Self {
        self.embedding_query_prefix = Some(embedding_query_prefix.to_string());
        self
    }

    pub fn catalog_path(mut self, catalog_path: &str) -> Self {
        self.catalog_path = Some(catalog_path.to_string());
        self
    }

    pub fn balanced_preference(mut self) -> Self {
        self.query_preference = Some("balanced".to_string());
        self
    }

    pub fn deterministic_preference(mut self) -> Self {
        self.query_preference = Some("deterministic".to_string());
        self
    }

    pub fn creative_preference(mut self) -> Self {
        self.query_preference = Some("creative".to_string());
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
        let catalog_path = match self.catalog_path {
            Some(catalog_path) => catalog_path,
            None => "./catalog".to_string(),
        };
        let catalog = match DeepThoughtVecStore::new(&catalog_path) {
            Ok(catalog) => catalog,
            Err(err) => bail!("Failed to create catalog: {}", err),
        };
        router.catalog = Some(catalog);
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
        router.query_preference = match self.query_preference {
            Some(preference) => preference,
            None => "balanced".to_string(),
        };
        Ok(router)
    }
}
