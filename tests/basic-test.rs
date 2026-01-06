#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
    use deepthought::*;

    #[test]
    fn test_new_backend() {
        let _ = DeepThoughtBackend::new().unwrap();
    }

    #[test]
    fn test_new_load_model() {
        let dtb = DeepThoughtBackend::new().unwrap();
        let _ = dtb.load_model("Llama-3.2-3B-Instruct-Q6_K.gguf", "You are robot!").unwrap();
    }

    #[test]
    fn test_new_who_are_you() {
        let dtb = DeepThoughtBackend::new().unwrap();
        let mut dtm = dtb.load_model("Llama-3.2-3B-Instruct-Q6_K.gguf", "You are robot!").unwrap();
        let res = dtm.chat("Who are you?").unwrap();
        assert!(res.to_lowercase().contains("model"));
    }

}
