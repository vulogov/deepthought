extern crate log;

use easy_error::bail;
use minijinja::{Environment, context};

use crate::*;
use vecstore::Neighbor;

impl DeepThoughtVecStore {
    pub fn register_template(
        &mut self,
        name: &str,
        template: &str,
    ) -> Result<(), easy_error::Error> {
        self.templates
            .insert(name.to_string(), template.to_string());
        Ok(())
    }

    pub fn output(&self, name: &str, n: Neighbor) -> Result<String, easy_error::Error> {
        let mut res: String = String::new();
        let env = Environment::new();
        let template = match env.get_template(name) {
            Ok(template) => template,
            Err(err) => bail!("Template not found: {}", err),
        };
        let text_str = match n.metadata.fields.get("text") {
            Some(text) => text.as_str(),
            None => bail!("Missing 'text' field in metadata"),
        };
        let context = context! {
            id => n.id,
            score => n.score,
            text => text_str,
        };
        match template.render(&context) {
            Ok(rendered) => res.push_str(&rendered),
            Err(err) => bail!("Error rendering template: {}", err),
        }
        Ok(res)
    }
}
