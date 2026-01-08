extern crate log;

use crate::*;

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
            dbpath: ".".to_string(),
            backend: backend,
            model: model,
            embed_model: None,
            embedding_doc_prefix: String::from(""),
            embedding_query_prefix: String::from(""),
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

    pub fn chat(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        self.model.chat(prompt)
    }

    pub fn ask(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        self.model.ask(prompt)
    }

    pub fn embed(&mut self, prompt: &str) -> Result<Vec<Vec<f32>>, easy_error::Error> {
        match self.embed_model {
            Some(ref mut model) => match model.embed(&[prompt]) {
                Ok(embeddings) => Ok(embeddings),
                Err(err) => easy_error::bail!("EMBED ERROR: {:?}", err),
            },
            None => easy_error::bail!("Embedding model not loaded"),
        }
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
