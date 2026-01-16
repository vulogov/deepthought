extern crate log;

use easy_error::bail;

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
            sessions: HashMap::new(),
        })
    }
}
