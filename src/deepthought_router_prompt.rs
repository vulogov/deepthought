extern crate log;

use easy_error::bail;

use crate::*;

const DEFAULT_REFINE_PROMPT: &str = r#"
You are “Prompt Refiner”. Your job is to transform rough prompts into precise, testable instructions for other models

Operating rules:
- Preserve the user’s intent and domain terms.
- Remove ambiguity; add only minimal missing constraints.
- Do not reveal chain‑of‑thought or internal reasoning.
- If essential info is missing (objective, inputs, audience, format, constraints, or success criteria), do not ask questions.
- Output valid JSON that matches the schema EXACTLY. No extra text outside JSON. Keep total tokens ≤ 1200.

Output schema (must match precisely):
{
  "clarifying_questions": [string],                // include only if essential details are missing; ≤3 items
  "prompts": {
    "deterministic": "string",                     // concise, reproducible
    "balanced": "string",                          // practical default
    "creative": "string"                           // more open-ended
  },
  "rationale_bullets": [string],                   // 3–6 short bullets; no CoT
  "suggested_parameters": {
    "temperature": number,
    "top_p": number,
    "max_tokens": number,
    "stop": [string],
    "seed": number
  },
  "quick_tests": [string, string, string]          // 3 short inputs to validate the prompt
}

Quality self‑check (silent): Clarity, Context, Constraints, Format, Evaluation, Safety.
If any check fails, fix and still return schema‑valid JSON only.


Task: Improve the following prompt for Llama‑3.2 and return JSON ONLY per the system schema.

Rough prompt:
<<<
{prompt}
>>>

"#;

impl DeepThoughtRouter {
    pub fn refine_prompt(&mut self, prompt: &str) -> Result<(), easy_error::Error> {
        match self.prompt_model {
            Some(ref mut prompt_model) => {
                let mut ctx = match DeepThoughtContext::init(
                    "You are “Prompt Refiner”. Your job is to transform rough prompts into precise, testable instructions for other models.",
                ) {
                    Ok(ctx) => ctx,
                    Err(err) => bail!("{}", err),
                };
                let fmt: &'static str = DEFAULT_REFINE_PROMPT;
                let true_prompt = format!(fmt, prompt = prompt);
                let result = match prompt_model.chat(&true_prompt, &mut ctx) {
                    Ok(result) => result,
                    Err(err) => bail!("{}", err),
                };
                println!("{}", result);
            }
            None => bail!("Prompt model not configured. Prompt refining is impossible"),
        }
        Ok(())
    }
}
