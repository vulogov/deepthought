extern crate log;

use easy_error::bail;

use crate::*;

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a helpful assistant."#;

impl DeepThoughtRouter {
    pub fn new_session(&mut self, name: &str) -> Result<(), easy_error::Error> {
        let new_context = match DeepThoughtContext::init(DEFAULT_SYSTEM_PROMPT) {
            Ok(context) => context,
            Err(err) => {
                bail!("CONTEXT ERROR: {:?}", err);
            }
        };
        self.sessions.insert(name.to_string(), new_context);
        Ok(())
    }
    pub fn get_session(
        &mut self,
        name: &str,
    ) -> Result<&mut DeepThoughtContext, easy_error::Error> {
        match self.sessions.get_mut(name) {
            Some(context) => Ok(context),
            None => bail!("Session not found"),
        }
    }
}
