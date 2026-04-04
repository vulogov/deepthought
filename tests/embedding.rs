// ============================================
// Example Usage and Test Helper
// ============================================

#[cfg(test)]
pub mod tests {
    use super::*;

    // Mock embedder for testing
    pub struct MockEmbedder;

    impl EmbeddingProvider for MockEmbedder {
        fn generate_embedding(&self, _text: &str, _prefix: &str) -> Result<Vec<f32>, Error> {
            // Return a dummy embedding vector of dimension 384
            Ok(vec![0.0; 384])
        }

        fn generate_batch_embeddings(
            &self,
            texts: &[String],
            _prefix: &str,
        ) -> Result<Vec<Vec<f32>>, Error> {
            Ok(vec![vec![0.0; 384]; texts.len()])
        }
    }

    #[test]
    fn test_external_embeddings() {
        let mut store = DeepThoughtVecStore::new("test.db").unwrap();
        let embedder = MockEmbedder;

        // Use the new method with external embeddings
        let result = store.add_document_with_external_embeddings(
            "test-doc",
            "This is a test document with multiple sentences. It should be split into chunks.",
            &embedder,
            false, // sequential mode
        );

        assert!(result.is_ok());
    }
}
