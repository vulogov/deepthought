extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtContext {
    pub fn new() -> Result<Self, easy_error::Error> {
        Ok(DeepThoughtContext {
            messages: Vec::new(),
            max_msg: None,
        })
    }
    pub fn init(system_prompt: &str) -> Result<Self, easy_error::Error> {
        let mut dtc = match DeepThoughtContext::new() {
            Ok(dtc) => dtc,
            Err(err) => bail!("{}", err),
        };
        match dtc.system(system_prompt) {
            Ok(_) => {}
            Err(err) => bail!("{}", err),
        }
        Ok(dtc)
    }
    pub fn maximum_messages(&mut self, max_msg: usize) {
        self.max_msg = Some(max_msg);
    }
    pub fn system(&mut self, system_prompt: &str) -> Result<(), easy_error::Error> {
        match self.max_msg {
            Some(max_msg) => {
                if self.messages.len() > max_msg {
                    bail!("Context is full")
                }
            }
            None => {}
        }
        let msg = match LlamaChatMessage::new("system".to_string(), system_prompt.to_string()) {
            Ok(msg) => msg,
            Err(err) => bail!("{}", err),
        };
        self.messages.push(msg);
        Ok(())
    }
    pub fn user(&mut self, prompt: &str) -> Result<(), easy_error::Error> {
        match self.max_msg {
            Some(max_msg) => {
                if self.messages.len() > max_msg {
                    bail!("Context is full")
                }
            }
            None => {}
        }
        let msg = match LlamaChatMessage::new("user".to_string(), prompt.to_string()) {
            Ok(msg) => msg,
            Err(err) => bail!("{}", err),
        };
        self.messages.push(msg);
        Ok(())
    }
    pub fn assistant(&mut self, prompt: &str) -> Result<(), easy_error::Error> {
        match self.max_msg {
            Some(max_msg) => {
                if self.messages.len() > max_msg {
                    bail!("Context is full")
                }
            }
            None => {}
        }
        let msg = match LlamaChatMessage::new("assistant".to_string(), prompt.to_string()) {
            Ok(msg) => msg,
            Err(err) => bail!("{}", err),
        };
        self.messages.push(msg);
        Ok(())
    }
    pub fn messages(&self) -> &Vec<LlamaChatMessage> {
        &self.messages
    }
    pub fn remove_last(&mut self) {
        let _ = self.messages.pop();
    }
}
