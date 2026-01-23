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
}
