extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtRouter {
    pub fn recommended_prompt(&mut self, prompt: &str) -> Result<String, easy_error::Error> {
        let refined_prompt = match self.refine_prompt(prompt) {
            Ok(refined_prompt) => refined_prompt,
            Err(err) => bail!("{}", err),
        };
        Ok(refined_prompt
            .recommended_prompt(&self.query_preference.clone())
            .to_string())
    }
    pub fn chat(&mut self, route_name: &str, query: &str) -> Result<String, easy_error::Error> {
        let actual_prompt = match self.recommended_prompt(query) {
            Ok(recommended_prompt) => recommended_prompt,
            Err(err) => bail!("{}", err),
        };
        let model = match self.get_route(route_name) {
            Some(model) => model,
            None => bail!("Route {} not found", route_name),
        };
        match model.chat(&actual_prompt) {
            Ok(result) => Ok(result.clone()),
            Err(err) => bail!("{}", err),
        }
    }
}
