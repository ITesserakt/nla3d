use std::ops::{Deref, DerefMut};
use nla3d_math::dense::{Mat, MatSym, Vec};
use nla3d_math::mat;

static LEVI_CIVITA: Vec<3, Mat<3, 3>> = mat![[
    mat![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ],
    mat![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    mat![
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
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
    #[inline]
    #[must_use]
    pub fn j(&self) -> f64 {
        self.0
    }
}
