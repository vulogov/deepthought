extern crate log;

use easy_error::bail;
use rust_rule_engine::RustRuleEngine;

use crate::*;

impl DeepThoughtRouter {
    pub fn expert_system(&mut self, rules: &str, facts: &str) -> Result<(), easy_error::Error> {
        let kb = (*self.knowledge_base).clone();
        let mut engine = RustRuleEngine::new(kb);
        let rules = match self.get_rules(rules) {
            Ok(rules) => rules,
            Err(e) => bail!("{}", e),
        };
        let facts = match self.get_facts(facts) {
            Ok(facts) => facts,
            Err(e) => bail!("{}", e),
        };
        for r in rules {
            match engine.knowledge_base().add_rule(r) {
                Ok(_) => (),
                Err(e) => bail!("Failed to add rule: {}", e),
            }
        }
        let _result = match engine.execute(facts) {
            Ok(result) => result,
            Err(e) => bail!("{}", e),
        };
        Ok(())
    }
}
