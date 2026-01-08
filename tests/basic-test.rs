#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
    use deepthought::*;
    use rust_dynamic::value::Value;

    #[test]
    fn test_new_backend() {
        let _ = DeepThoughtBackend::new().unwrap();
    }

    #[test]
    fn test_new_load_model() {
        let dtb = DeepThoughtBackend::new().unwrap();
        let _ = dtb
            .load_model(&std::env::var("LLAMATEST_GGUF").unwrap(), "You are robot!")
            .unwrap();
    }

    #[test]
    fn test_new_who_are_you() {
        let dtb = DeepThoughtBackend::new().unwrap();
        let mut dtm = dtb
            .load_model(&std::env::var("LLAMATEST_GGUF").unwrap(), "You are robot!")
            .unwrap();
        let res = dtm.chat("Who are you?").unwrap();
        assert!(res.to_lowercase().contains("model"));
    }

    #[test]
    fn test_new_who_are_you_deepthough() {
        let mut dt = DeepThought::new(&std::env::var("LLAMATEST_GGUF").unwrap()).unwrap();
        let res = dt.chat("Who are you?").unwrap();
        assert!(res.to_lowercase().contains("model"));
    }

    #[test]
    fn test_load_model_and_embed() {
        let dtb = DeepThoughtBackend::new().unwrap();
        let mut dtm = dtb
            .load_model(&std::env::var("LLAMATEST_GGUF").unwrap(), "You are robot!")
            .unwrap();
        let emb = dtm.embed(&["Hello world!"]).unwrap();
        println!("{:?}", emb);
    }
}
