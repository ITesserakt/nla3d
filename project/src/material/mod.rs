use platform::collections::hash_map::HashMap;
use platform::collections::string::String;

pub struct Material {
    name: String,
    constants: HashMap<String, f64>
}

impl Material {
    pub fn new() -> Self {
        todo!()
    }
}
