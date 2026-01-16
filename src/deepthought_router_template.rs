extern crate log;

use easy_error::bail;
use minijinja::Environment;

use crate::*;

impl DeepThoughtRouter {
    pub fn template(
        raw_template: &str,
        context: minijinja::Value,
    ) -> Result<String, easy_error::Error> {
        let mut env = Environment::new();
        match env.add_template("main", raw_template) {
            Ok(_) => (),
            Err(err) => bail!("Error adding template: {}", err),
        }
        let template = match env.get_template("main") {
            Ok(template) => template,
            Err(err) => bail!("Template not found: {}", err),
        };
        match template.render(&context) {
            Ok(rendered) => Ok(rendered),
            Err(err) => bail!("Error rendering template: {}", err),
        }
    }
}
