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
}
