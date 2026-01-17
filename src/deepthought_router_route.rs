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

    pub fn get_route(&mut self, name: &str) -> Option<&mut DeepThought> {
        self.routes.get_mut(name)
    }
}
