extern crate log;

use easy_error::bail;
use rust_rule_engine::GRLParser;

use crate::*;

impl DeepThoughtRouter {
    pub fn new_rules(&mut self, name: &str, value: &str) -> Result<(), easy_error::Error> {
        let rules = match GRLParser::parse_rules(value) {
            Ok(rules) => rules,
            Err(e) => bail!("Failed to parse rules: {}", e),
        };
        self.rules.insert(name.to_string(), rules);
        Ok(())
    }
    pub fn get_rules(&mut self, name: &str) -> Result<Vec<Rule>, easy_error::Error> {
        match self.rules.get(name) {
            Some(rules) => Ok(rules.clone()),
            None => bail!("No rules found for {}", name),
        }
    }
}
