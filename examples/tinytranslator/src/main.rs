use deepthought::DeepThought;


static SYSTEM_PROMPT: &str = r#"You are a translator from SimpleLang to English.
Rules:
- Output ONLY the English translation, no explanations.
- If input is not SimpleLang, reply with "Untranslatable".
Examples:
- Мне нравится хлеб .  → I love bread.
- Ты меня видишь ?  → Do you see me?
- Мне не нравится вода . → I do not love water.
"#;


fn main() {
    let mut dt = DeepThought::new(&std::env::var("LLMTEST_GGUF").unwrap()).unwrap();
    let output = dt
        .chat(SYSTEM_PROMPT)
        .unwrap();
    println!("{}", output);
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    let src_txt = "Мне нравится вода".to_string();
    println!("Translating: {}", src_txt);
    let output = dt
        .chat(&src_txt)
        .unwrap();
    println!("{}", output);
}
