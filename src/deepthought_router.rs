extern crate log;

use easy_error::bail;
use rust_rule_engine::KnowledgeBase;

use crate::*;

impl DeepThoughtRouter {
    pub fn new() -> Result<Self, easy_error::Error> {
        let backend = match DeepThoughtBackend::new() {
            Ok(backend) => backend,
            Err(err) => {
                bail!("BACKEND ERROR: {:?}", err);
            }
        };
        Ok(DeepThoughtRouter {
            backend: backend,
            routes: HashMap::new(),
            ctx_routes: HashMap::new(),
            sessions: HashMap::new(),
            facts: HashMap::new(),
            embed_model: None,
            prompt_model: None,
            catalog: None,
            query_preference: "balanced".to_string(),
            embedding_query_prefix: "".to_string(),
            knowledge_base: Arc::new(KnowledgeBase::new(&nanoid::nanoid!())),
        })
    }
    pub fn embed_model(&mut self, gguf_model: &str) -> Result<(), easy_error::Error> {
        let model = match self.backend.load_model(gguf_model, "You are the robot!") {
            Ok(model) => model,
            Err(err) => {
                easy_error::bail!("EMBED MODEL ERROR: {:?}", err);
            }
        };
        self.embed_model = Some(model);
        Ok(())
    }
}
