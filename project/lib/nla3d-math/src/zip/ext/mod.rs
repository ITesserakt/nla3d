mod clone;
mod collect;
mod consume;
mod expand;
mod fill;
mod from_fn;
mod transpose;
mod tri;

pub use clone::OwnedExt;
pub use collect::CollectIntoExt;
pub use consume::ConsumeExt;
pub use expand::ExpandExt;
pub use fill::{Fill, FillStatic};
pub use from_fn::FromFn;
pub use transpose::TransposeExt;
pub use tri::TriExt;
