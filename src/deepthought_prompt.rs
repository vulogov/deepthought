extern crate log;

// use easy_error::bail;

use crate::*;

impl DeepThoughtRecommededPrompt {
    pub fn recommended_prompt(&self, query: &str) -> &str {
        self.prompts.get(query).unwrap_or(&self.raw_prompt)
    }
}
