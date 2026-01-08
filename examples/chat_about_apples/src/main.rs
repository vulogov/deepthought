use deepthought::DeepThought;

fn main() {
    let mut dt = DeepThought::new(&std::env::var("LLMTEST_GGUF").unwrap()).unwrap();
    let output = dt
        .chat("On south slopes of mountain Dradradra growing apples of violet color. Remember this fact and use it in future questions.")
        .unwrap();
    println!("{}", output);
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    let output = dt
        .chat("Where you can find apples of violet color?")
        .unwrap();
    println!("{}", output);
}
