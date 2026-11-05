#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate core as std;

pub mod element;
pub mod dof;
pub mod material;
pub mod solidmech;
