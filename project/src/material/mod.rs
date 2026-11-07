use platform::collections::hash_map::HashMap;
use platform::compat::string::String;

pub struct Material {
    name: String,
    constants: HashMap<String, f64>
}

impl Material {
    pub fn new() -> Self {
        todo!()
    }
}
