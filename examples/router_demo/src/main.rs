use deepthought::DeepThoughtRouterBuilder;

fn main() {
    let mut dtr = DeepThoughtRouterBuilder::default()
        .default_embed_model("../../nomic-embed-text-v1.Q5_K_M.gguf")
        .prompt_model("../../Llama-3.2-3B-Instruct-Q6_K.gguf")
        .embedding_query_prefix("search_query")
        .build()
        .unwrap();
    let recommended_prompt = dtr.refine_prompt("Who is Anna Karenina?").unwrap();
    println!("{:?}", &recommended_prompt);
}
