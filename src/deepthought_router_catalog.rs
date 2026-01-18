extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtRouter {
    pub fn add_url_to_catalog(&mut self, doc: &str, url: &str) -> Result<(), easy_error::Error> {
        let mut obj = Value::from_str(doc);
        obj.set_tag("url", url);
        obj.set_tag("type", "url");
        self.add_object_to_catalog(obj)
    }
    pub fn add_endpoint_to_catalog(
        &mut self,
        endpoint_type: &str,
        doc: &str,
        url: &str,
    ) -> Result<(), easy_error::Error> {
        let mut obj = Value::from_str(doc);
        obj.set_tag("url", url);
        obj.set_tag("endpoint_type", endpoint_type);
        obj.set_tag("type", "endpoint");
        self.add_object_to_catalog(obj)
    }
    pub fn add_route_to_catalog(
        &mut self,
        doc: &str,
        route: &str,
    ) -> Result<(), easy_error::Error> {
        let mut obj = Value::from_str(doc);
        obj.set_tag("route", route);
        obj.set_tag("type", "route");
        self.add_object_to_catalog(obj)
    }

    pub fn add_object_to_catalog(&mut self, obj: Value) -> Result<(), easy_error::Error> {
        let embedder = match &self.embed_model {
            Some(embed_model) => embed_model,
            None => bail!("Embedding model not set"),
        };
        match self.catalog {
            Some(ref mut catalog) => match catalog.add_object(&obj.id.clone(), obj, &embedder) {
                Ok(_) => Ok(()),
                Err(err) => bail!("Error adding value: {}", err),
            },
            None => bail!("Vector store not set"),
        }
    }
}
