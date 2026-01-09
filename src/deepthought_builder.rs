extern crate log;
use crate::{DeepThought, DeepThoughtBuilder, DeepThoughtVecStore};
use easy_error::bail;
use grainfs::dir::create_dir_recursive;
use grainfs::path::*;

use crate::deepthought_backend::{DEFAULT_BATCH_SIZE, DEFAULT_CONTEXT_LENGTH};
use crate::deepthought_vector::{DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE};

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
            chunk_size: Some(DEFAULT_CHUNK_SIZE),
            chunk_overlap: Some(DEFAULT_CHUNK_OVERLAP),
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

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    pub fn chunk_overlap(mut self, size: usize) -> Self {
        self.chunk_overlap = Some(size);
        self
    }

    fn fix_the_path(path: String) -> Option<String> {
        match try_expand_vars(&path) {
            Some(expanded_path) => match normalize_path(&expanded_path) {
                Ok(normalized_path) => {
                    let normalized_path_str = normalized_path.display().to_string();
                    if path_exists(&normalized_path_str) {
                        Some(normalized_path_str)
                    } else {
                        match create_dir_recursive(&normalized_path_str) {
                            Ok(_) => Some(normalized_path_str),
                            Err(err) => {
                                log::error!("Failed to create directory: {:?}", err);
                                return None;
                            }
                        }
                    }
                }
                Err(_) => {
                    return None;
                }
            },
            None => {
                return None;
            }
        }
    }

    pub fn build(self) -> Result<DeepThought, easy_error::Error> {
        let dbpath = match self.dbpath {
            Some(ref path) => path,
            None => &String::from("db"),
        };
        if !path_exists(&dbpath) {
            log::debug!("Creating new telemetry bucket at {}", &dbpath);
            let fixed_path = match DeepThoughtBuilder::fix_the_path(dbpath.clone()) {
                Some(path) => path,
                None => bail!("ERROR fixing telemetry bucket path"),
            };
            match create_dir_recursive(&fixed_path) {
                Ok(_) => {}
                Err(err) => {
                    bail!("Failed to create DeepThought database directory: {:?}", err);
                }
            }
        }
        let chat_gguf = match self.chat_model_gguf {
            Some(chat_path) => chat_path,
            None => String::from("chat.gguf"),
        };
        let mut model = match DeepThought::new(&chat_gguf) {
            Ok(model) => model,
            Err(err) => bail!("ERROR creating chat model: {}", err),
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
        let chunk_size = match self.chunk_size {
            Some(size) => size,
            None => DEFAULT_CHUNK_SIZE,
        };
        let chunk_overlap = match self.chunk_overlap {
            Some(size) => size,
            None => DEFAULT_CHUNK_OVERLAP,
        };
        let mut vecstore = match DeepThoughtVecStore::new(dbpath) {
            Ok(vecstore) => vecstore,
            Err(err) => bail!("ERROR opening vector store: {}", err),
        };
        vecstore.chunk_size = chunk_size;
        vecstore.chunk_overlap = chunk_overlap;
        vecstore.embedding_prefix = self.embedding_doc_prefix.clone();
        model.model.context_length = context_len;
        model.model.batch_size = batch_size;
        model.vecstore = Some(vecstore);
        Ok(model)
    }
}
