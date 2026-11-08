use nla3d_math::{
    dense::{Mat, MatSym, Vec},
    m, v,
    zip::Zip,
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
        use num_traits::Float;

        self.determinant().sqrt()
    }

    #[inline]
    pub fn inv(&self) -> MatSym<3> {
        self.0.inv()
    }

    #[inline]
    #[allow(non_snake_case)]
    pub fn E(&self) -> MatSym<3> {
        Zip::from(&self.0)
            .and(&MatSym::<_, f64>::identity())
            .map(|c, i| (c - i) * 0.5)
    }

    #[inline]
    pub fn invariants(&self) -> Vec<3> {
        v![
            self.xx + self.yy + self.zz,
            self.xx * self.yy + self.yy * self.zz + self.xx * self.zz
                - self.xy * self.xy
                - self.yz * self.yz
                - self.xz * self.xz,
            self.determinant()
        ]
    }
}
