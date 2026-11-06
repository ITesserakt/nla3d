use nla3d_math::{
    dense::{Mat, MatSym, Vec},
    m,
};
use std::ops::{Deref, DerefMut};

static LEVI_CIVITA: Vec<3, Mat<3, 3>> = m![[
    m![[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    m![[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],],
    m![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
]];

#[derive(Debug)]
pub struct CauchyStrainTensor(MatSym<3>);

impl Deref for CauchyStrainTensor {
    type Target = MatSym<3>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CauchyStrainTensor {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl CauchyStrainTensor {
    #[allow(non_snake_case)]
    #[inline]
    pub fn J(&self) -> f64 {
        #[cfg(not(feature = "std"))]
        use nla3d_math::num_traits::Float;
        
        self.0.determinant().sqrt()
    }
}
