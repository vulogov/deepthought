extern crate log;
use crate::{DeepThought, DeepThoughtBuilder};
use easy_error::bail;

use crate::deepthought_backend::{DEFAULT_BATCH_SIZE, DEFAULT_CONTEXT_LENGTH};

impl DeepThoughtBuilder {
    pub fn new() -> Self {
        DeepThoughtBuilder {
            dbpath: None,
            chat_model_gguf: None,
            embed_model_gguf: None,
            context_length: None,
            batch_size: None,
            embedding_doc_prefix: String::from(""),
            embedding_query_prefix: String::from(""),
        }
    }

    pub fn chat_model_gguf(mut self, path: String) -> Self {
        self.chat_model_gguf = Some(path);
        self
    }

    pub fn embed_model_gguf(mut self, path: String) -> Self {
        self.embed_model_gguf = Some(path);
        self
    }

    pub fn dbpath(mut self, path: String) -> Self {
        self.dbpath = Some(path);
        self
    }

    pub fn embedding_doc_prefix(mut self, prefix: String) -> Self {
        self.embedding_doc_prefix = prefix;
        self
    }

    pub fn embedding_query_prefix(mut self, prefix: String) -> Self {
        self.embedding_query_prefix = prefix;
        self
    }

    pub fn context_length(mut self, length: usize) -> Self {
        self.context_length = Some(length);
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn build(self) -> Result<DeepThought, easy_error::Error> {
        let dbpath = match self.dbpath {
            Some(path) => path,
            None => String::from("db"),
        };
        let chat_gguf = match self.chat_model_gguf {
            Some(path) => path,
            None => String::from("chat.gguf"),
        };
        let mut model = match DeepThought::new(&chat_gguf) {
            Ok(model) => model,
            Err(err) => bail!("ERROR creat8ing chat model: {}", err),
        };
        match self.embed_model_gguf {
            Some(path) => match model.embed_model(&path) {
                Ok(_) => {}
                Err(err) => bail!("ERROR creating embedding model: {}", err),
            },
            None => {
                log::debug!("Embedding model not provided");
            }
        };
        let context_len = match self.context_length {
            Some(len) => len,
            None => DEFAULT_CONTEXT_LENGTH,
        };
        let batch_size = match self.batch_size {
            Some(size) => size,
            None => DEFAULT_BATCH_SIZE,
        };
        model.dbpath = dbpath;
        model.model.context_length = context_len;
        model.model.batch_size = batch_size;
        Ok(model)
    }
}
