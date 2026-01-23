extern crate log;

use easy_error::bail;
use rust_rule_engine::Facts;
use rust_rule_engine::Value as RREValue;

use crate::*;

impl DeepThoughtRouter {
    pub fn new_fact(
        &mut self,
        name: &str,
        key: &str,
        value: RREValue,
    ) -> Result<(), easy_error::Error> {
        let facts = match self.facts.get_mut(name) {
            Some(facts) => facts,
            None => bail!("No facts found for {}", name),
        };
        facts.set(key, value);
        Ok(())
    }
    pub fn get_facts(&mut self, name: &str) -> Result<&mut Facts, easy_error::Error> {
        match self.facts.get_mut(name) {
            Some(facts) => Ok(facts),
            None => bail!("No facts found for {}", name),
        }
    }
    pub fn facts_collection(&mut self, name: &str) -> Result<&mut Facts, easy_error::Error> {
        Ok(self
            .facts
            .entry(name.to_string())
            .or_insert_with(|| Facts::new()))
    }
}
