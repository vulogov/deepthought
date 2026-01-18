extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtRouter {
    pub fn chat(&mut self, route_name: &str, query: &str) -> Result<String, easy_error::Error> {
        let refined_prompt = match self.refine_prompt(query) {
            Ok(refined_prompt) => refined_prompt,
            Err(err) => bail!("{}", err),
        };
        let actual_prompt = refined_prompt.recommended_prompt(&self.query_preference);
        let model = match self.get_route(route_name) {
            Some(model) => model,
            None => bail!("Route {} not found", route_name),
        };
        match model.chat(actual_prompt) {
            Ok(result) => Ok(result.clone()),
            Err(err) => bail!("{}", err),
        }
    }
}
