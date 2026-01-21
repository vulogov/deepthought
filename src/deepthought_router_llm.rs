extern crate log;

use easy_error::bail;

use crate::*;

impl DeepThoughtRouter {
    pub fn query_preference(&self) -> &str {
        self.query_preference.as_str()
    }

    pub fn set_query_preference(&mut self, preference: String) {
        self.query_preference = preference;
    }

    pub fn query(
        &mut self,
        route_name: &str,
        query: &str,
        template_name: &str,
    ) -> Result<HashMap<String, Value>, easy_error::Error> {
	let actual_prompt = match self.recommended_prompt(&query) {
            Ok(actual_prompt) => actual_prompt,
            Err(err) => bail!("{}", err),
        };
        let router_obj = match self.get_route(route_name) {
            Some(router_obj) => router_obj,
            None => bail!("Router {} not found", &route_name),
        };
        let res = match router_obj.query_templated(&actual_prompt, template_name) {
            Ok(res) => res,
            Err(err) => bail!("{}", err),
        };
        Ok(res)
    }
}
