extern crate log;

use easy_error::bail;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use took::Timer;

use crate::*;
use vecstore::text_splitter::RecursiveCharacterTextSplitter;
use vecstore::{HybridQuery, TextSplitter};
use vecstore::{Metadata, Neighbor, VecStore};

pub const DEFAULT_CHUNK_SIZE: usize = 1024;
pub const DEFAULT_CHUNK_OVERLAP: usize = 128;
pub const DEFAULT_K: usize = 10;
pub const DEFAULT_ALPHA: f32 = 0.7;

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
            embedding_prefix: "".to_string(),
            k: DEFAULT_K,
            alpha: DEFAULT_ALPHA,
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
        embedder: &DeepThoughtModel,
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
            let vector = match embedder.embed(&[format!("{} {}", self.embedding_prefix, c)]) {
                Ok(vector) => vector[0].clone(),
                Err(err) => bail!("Failed to embed text: {:?}", err),
            };
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields.insert("id".into(), serde_json::json!(id));
            meta.fields.insert("n".into(), serde_json::json!(n));
            meta.fields.insert("text".into(), serde_json::json!(text));
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
    fn query_neighbors(
        &self,
        embedding: Vec<f32>,
        query: &str,
    ) -> Result<Vec<Neighbor>, easy_error::Error> {
        let conn = self.conn.clone();
        let conn_read = match conn.read() {
            Ok(conn_read) => conn_read,
            Err(err) => {
                bail!("Failed to acquire read lock: {:?}", err);
            }
        };
        let h_query = HybridQuery {
            vector: embedding,
            keywords: query.to_string(),
            filter: None,
            k: self.k,
            alpha: self.alpha,
        };
        let results = match conn_read.hybrid_query(h_query) {
            Ok(results) => results,
            Err(err) => {
                bail!("Failed to query vector store: {:?}", err);
            }
        };
        drop(conn_read);
        drop(conn);
        Ok(results)
    }

    pub fn query(
        &self,
        embedding: Vec<f32>,
        query: &str,
    ) -> Result<Vec<String>, easy_error::Error> {
        let mut res: Vec<String> = Vec::new();
        let neighbors = match self.query_neighbors(embedding, query) {
            Ok(neighbors) => neighbors,
            Err(err) => {
                bail!("Failed to query vector store: {:?}", err);
            }
        };
        for neighbor in neighbors.iter() {
            match neighbor.metadata.fields.get("text") {
                Some(text) => res.push(text.to_string()),
                None => {}
            }
        }
        Ok(res)
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
