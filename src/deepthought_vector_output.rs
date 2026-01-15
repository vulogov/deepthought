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
        let raw_template: &str = match self.templates.get(name) {
            Some(template) => template,
            None => bail!("Template not found: {}", name),
        };
        let mut env = Environment::new();
        match env.add_template(name, raw_template) {
            Ok(_) => (),
            Err(err) => bail!("Error adding template: {}", err),
        };
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
    pub fn query_templated(
        &self,
        embedding: Vec<f32>,
        template_name: &str,
        query: &str,
    ) -> Result<HashMap<String, Value>, easy_error::Error> {
        let mut res: HashMap<String, Value> = HashMap::new();
        res.insert("query".to_string(), Value::from_string(query.to_string()));
        let neighbors = match self.query_neighbors(embedding, query) {
            Ok(neighbors) => neighbors,
            Err(err) => {
                bail!("Failed to query vector store: {:?}", err);
            }
        };
        let mut rag = Value::list();
        for neighbor in neighbors.iter() {
            let formatted_text = match self.output(template_name, neighbor.clone()) {
                Ok(formatted_text) => formatted_text,
                Err(err) => {
                    bail!("Failed to render template: {:?}", err);
                }
            };
            rag = rag.push(Value::from_string(formatted_text));
        }
        res.insert("rag".to_string(), rag);
        Ok(res)
    }
}
