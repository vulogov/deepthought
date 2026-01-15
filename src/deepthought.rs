extern crate log;

use easy_error::bail;

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
            vecstore: None,
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
    pub fn add_document(&mut self, doc: &str) -> Result<(), easy_error::Error> {
        let embedder = match &self.embed_model {
            Some(embed_model) => embed_model,
            None => bail!("Embedding model not set"),
        };
        match self.vecstore {
            Some(ref mut vecstore) => {
                match vecstore.add_document(&nanoid::nanoid!(), doc, &embedder) {
                    Ok(_) => Ok(()),
                    Err(err) => bail!("Error adding document: {}", err),
                }
            }
            None => bail!("Vector store not set"),
        }
    }
    pub fn add_string(&mut self, doc: &str) -> Result<(), easy_error::Error> {
        let embedder = match &self.embed_model {
            Some(embed_model) => embed_model,
            None => bail!("Embedding model not set"),
        };
        match self.vecstore {
            Some(ref mut vecstore) => {
                match vecstore.add_string(&nanoid::nanoid!(), doc, &embedder) {
                    Ok(_) => Ok(()),
                    Err(err) => bail!("Error adding string: {}", err),
                }
            }
            None => bail!("Vector store not set"),
        }
    }
    pub fn add_value(&mut self, doc: Value) -> Result<(), easy_error::Error> {
        let embedder = match &self.embed_model {
            Some(embed_model) => embed_model,
            None => bail!("Embedding model not set"),
        };
        let value_str = match doc.conv(STRING) {
            Ok(value_str) => match value_str.cast_string() {
                Ok(value_casted) => value_casted,
                Err(err) => bail!("{}", err),
            },
            Err(err) => bail!("{}", err),
        };
        match self.vecstore {
            Some(ref mut vecstore) => match vecstore.add_string(&doc.id, &value_str, &embedder) {
                Ok(_) => Ok(()),
                Err(err) => bail!("Error adding value: {}", err),
            },
            None => bail!("Vector store not set"),
        }
    }
    pub fn delete_value(&mut self, doc: Value) -> Result<(), easy_error::Error> {
        match self.vecstore {
            Some(ref mut vecstore) => match vecstore.delete_record(&doc.id) {
                Ok(_) => Ok(()),
                Err(err) => bail!("Error deleting value: {}", err),
            },
            None => bail!("Vector store not set"),
        }
    }
    pub fn delete_record(&mut self, id: &str) -> Result<(), easy_error::Error> {
        match self.vecstore {
            Some(ref mut vecstore) => match vecstore.delete_record(id) {
                Ok(_) => Ok(()),
                Err(err) => bail!("Error deleting record: {}", err),
            },
            None => bail!("Vector store not set"),
        }
    }
    pub fn add_document_with_sync(&mut self, doc: &str) -> Result<(), easy_error::Error> {
        match self.add_document(doc) {
            Ok(_) => match self.sync() {
                Ok(_) => Ok(()),
                Err(err) => bail!("{}", err),
            },
            Err(err) => bail!("{}", err),
        }
    }
    pub fn add_string_with_sync(&mut self, doc: &str) -> Result<(), easy_error::Error> {
        match self.add_string(doc) {
            Ok(_) => match self.sync() {
                Ok(_) => Ok(()),
                Err(err) => bail!("{}", err),
            },
            Err(err) => bail!("{}", err),
        }
    }
    pub fn add_value_with_sync(&mut self, doc: Value) -> Result<(), easy_error::Error> {
        match self.add_value(doc) {
            Ok(_) => match self.sync() {
                Ok(_) => Ok(()),
                Err(err) => bail!("{}", err),
            },
            Err(err) => bail!("{}", err),
        }
    }
    pub fn query(&mut self, q: &str) -> Result<Vec<String>, easy_error::Error> {
        let embedder = match &self.embed_model {
            Some(embed_model) => embed_model,
            None => bail!("Embedding model not set"),
        };
        let query = format!("{} {}", self.embedding_query_prefix, q);
        let vector = match embedder.embed(&[query]) {
            Ok(vector) => vector,
            Err(err) => bail!("Error embedding query: {:?}", err),
        };
        let mut results: Vec<String> = match self.vecstore {
            Some(ref mut vecstore) => match vecstore.query(vector[0].clone(), q) {
                Ok(results) => results,
                Err(err) => bail!("Error adding document: {}", err),
            },
            None => bail!("Vector store not set"),
        };
        match self.chat(q) {
            Ok(res) => results.push(res),
            Err(err) => bail!("Error chatting: {}", err),
        }
        Ok(results)
    }
    pub fn len(&self) -> usize {
        match &self.vecstore {
            Some(vecstore) => match vecstore.len() {
                Ok(count) => count,
                Err(_) => 0,
            },
            None => 0,
        }
    }
    pub fn register_template(
        &mut self,
        name: &str,
        template: &str,
    ) -> Result<(), easy_error::Error> {
        match self.vecstore {
            Some(ref mut vecstore) => match vecstore.register_template(name, template) {
                Ok(_) => Ok(()),
                Err(err) => bail!("Error registering template: {}", err),
            },
            None => bail!("Vector store not set"),
        }
    }
    pub fn sync(&mut self) -> Result<(), easy_error::Error> {
        match &self.vecstore {
            Some(vecstore) => vecstore.save_vectorstore(),
            None => Ok(()),
        }
    }
}
