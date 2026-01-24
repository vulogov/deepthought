extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtRouter {
    pub fn query_vecstore(
        &mut self,
        route_name: &str,
        query: &str,
    ) -> Result<Vec<String>, easy_error::Error> {
        let model = match self.get_route(route_name) {
            Some(model) => model,
            None => bail!("Route {} not found", route_name),
        };
        match model.query_vecstore(query) {
            Ok(result) => Ok(result.clone()),
            Err(err) => bail!("{}", err),
        }
    }
    pub fn query_vecstore_templated(
        &mut self,
        route_name: &str,
        template_name: &str,
        query: &str,
    ) -> Result<HashMap<String, Value>, easy_error::Error> {
        let model = match self.get_route(route_name) {
            Some(model) => model,
            None => bail!("Route {} not found", route_name),
        };
        match model.query_vecstore_templated(query, template_name) {
            Ok(result) => Ok(result.clone()),
            Err(err) => bail!("{}", err),
        }
    }
    pub fn rag(
        &mut self,
        route_name: &str,
        template_name: &str,
        query: &str,
    ) -> Result<String, easy_error::Error> {
        let vec_res: HashMap<String, Value> =
            match self.query_vecstore_templated(route_name, template_name, query) {
                Ok(vec_res) => vec_res,
                Err(err) => bail!("{}", err),
            };
        let rag_data: Vec<Value> = match vec_res.get("rag") {
            Some(rag_data) => match rag_data.cast_list() {
                Ok(rag_data) => rag_data,
                Err(err) => bail!("{}", err),
            },
            None => bail!("No rag data is returned"),
        };
        let model = match self.get_route(route_name) {
            Some(model) => model,
            None => bail!("Route {} not found", route_name),
        };
        for r in rag_data.iter() {
            let rag_str = match r.conv(STRING) {
                Ok(rag_str) => match rag_str.cast_string() {
                    Ok(rag_str) => rag_str,
                    Err(err) => bail!("Error converting rag data to string: {:?}", err),
                },
                Err(err) => bail!("Error casting rag data to string: {:?}", err),
            };
            match model.add_inference_to_prompt(&rag_str) {
                Ok(_) => {}
                Err(err) => bail!("{}", err),
            }
        }
        self.chat(route_name, query)
    }
}
