#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
    use deepthought::*;
    use rust_dynamic::value::Value;

    #[test]
    fn test_new_who_are_you_deepthough_value() {
        let mut dt = DeepThought::new("Llama-3.2-3B-Instruct-Q6_K.gguf").unwrap();
        let res = dt.c(Value::from_str("Who are you?")).unwrap();
        assert!(res.cast_string().unwrap().to_lowercase().contains("model"));
    }

}
