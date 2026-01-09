extern crate log;

use easy_error::bail;
use std::sync::{Arc, RwLock};

use crate::*;
use vecstore::VecStore;

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
        };
        Ok(vector)
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
