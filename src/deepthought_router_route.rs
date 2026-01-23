extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtRouter {
    pub fn new_route(
        &mut self,
        name: &str,
        config: DeepThoughtBuilder,
    ) -> Result<(), easy_error::Error> {
        let new_route = match config.build() {
            Ok(new_route) => new_route,
            Err(err) => {
                bail!("ROUTE ERROR: {:?}", err);
            }
        };
        let _ = self.routes.insert(name.to_string(), new_route);
        Ok(())
    }

    pub fn new_ctx_route(
        &mut self,
        name: &str,
        ctx_route: DeepThoughtCtxModel,
    ) -> Result<(), easy_error::Error> {
        let _ = self.ctx_routes.insert(name.to_string(), ctx_route);
        Ok(())
    }

    pub fn get_route(&mut self, name: &str) -> Option<&mut DeepThought> {
        self.routes.get_mut(name)
    }
    pub fn get_ctx_route(&mut self, name: &str) -> Option<&mut DeepThoughtCtxModel> {
        self.ctx_routes.get_mut(name)
    }
    pub fn list_routes(&mut self) -> Vec<String> {
        self.routes.keys().cloned().collect()
    }
    pub fn list_ctx_routes(&mut self) -> Vec<String> {
        self.ctx_routes.keys().cloned().collect()
    }
    pub fn drop_route(&mut self, name: &str) -> Result<(), easy_error::Error> {
        match self.routes.remove(name) {
            Some(_) => Ok(()),
            None => bail!("Route not found"),
        }
    }
    pub fn drop_ctx_route(&mut self, name: &str) -> Result<(), easy_error::Error> {
        match self.ctx_routes.remove(name) {
            Some(_) => Ok(()),
            None => bail!("Route not found"),
        }
    }
}
