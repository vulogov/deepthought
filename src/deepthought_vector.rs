extern crate log;

use easy_error::bail;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use took::Timer;

use crate::*;
use vecstore::TextSplitter;
use vecstore::text_splitter::RecursiveCharacterTextSplitter;
use vecstore::{Metadata, VecStore};

pub const DEFAULT_CHUNK_SIZE: usize = 1024;
pub const DEFAULT_CHUNK_OVERLAP: usize = 128;

impl DeepThoughtVecStore {
    pub fn new(path: &str) -> Result<Self, easy_error::Error> {
        let conn = match VecStore::open(path) {
            Ok(conn) => conn,
            Err(err) => bail!("Failed to open vector store: {}", err),
        };
        let tvs: DeepThoughtVector = Arc::new(RwLock::new(conn));
        let vector = Self {
            path: Some(path.to_string()),
            conn: tvs,
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
        };
        Ok(vector)
    }
    pub fn split_text(&self, text: &str) -> Vec<String> {
        let splitter = RecursiveCharacterTextSplitter::new(self.chunk_size, self.chunk_overlap);
        let chunks: Vec<String> = match splitter.split_text(text) {
            Ok(chunks) => chunks,
            Err(err) => {
                log::debug!("Failed to split text: {}", err);
                Vec::new()
            }
        };
        chunks
    }
    pub fn add_document(
        &mut self,
        id: &str,
        text: &str,
        embedder: &mut DeepThoughtModel,
    ) -> Result<Duration, easy_error::Error> {
        let timer = Timer::new();
        let chunks: Vec<String> = self.split_text(text);
        let vectors = self.conn.clone();
        let mut conn = match vectors.write() {
            Ok(conn) => conn,
            Err(err) => bail!("Failed to acquire write lock: {}", err),
        };
        let mut n = 0;
        for c in chunks.iter() {
            let vector = match embedder.embed(&[c]) {
                Ok(vector) => vector[0].clone(),
                Err(err) => bail!("Failed to embed text: {:?}", err),
            };
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields.insert("id".into(), serde_json::json!(id));
            meta.fields.insert("n".into(), serde_json::json!(n));
            match conn.upsert(id.into(), vector.to_vec(), meta) {
                Ok(_) => {}
                Err(err) => bail!("Failed to add document: {}", err),
            };
            match conn.index_text(id.into(), c) {
                Ok(_) => {}
                Err(err) => bail!("Failed to index document: {}", err),
            };
            n += 1;
        }
        drop(conn);
        drop(vectors);
        let t = timer.took();
        let duration = t.as_std();
        Ok(*duration)
    }
    pub fn save_vectorstore(&self) -> Result<(), easy_error::Error> {
        let conn = self.conn.clone();
        let conn_write = match conn.write() {
            Ok(conn_write) => conn_write,
            Err(err) => {
                bail!("Failed to acquire write lock: {:?}", err);
            }
        };
        match conn_write.save() {
            Ok(_) => {}
            Err(err) => {
                bail!("Failed to save telemetry database: {:?}", err);
            }
        }
        drop(conn_write);
        drop(conn);
        Ok(())
    }
}
