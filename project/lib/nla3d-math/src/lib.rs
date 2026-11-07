#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "proptest")]
pub mod arbitrary;
pub mod dense;
pub mod sparse;
pub mod zip;

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate core as std;
#[cfg(not(feature = "std"))]
extern crate platform;

pub use faer::traits::num_traits;

#[cfg(feature = "compare")]
pub struct FaerCompare<T>(pub T);

#[cfg(feature = "compare")]
impl<T, V: Clone> matrixcompare::Matrix<V> for FaerCompare<T>
where
    T: faer::mat::AsMatRef<T = V>,
{
    #[inline]
    fn rows(&self) -> usize {
        use faer::Unbind;

        self.0.as_mat_ref().nrows().unbound()
    }

    #[inline]
    fn cols(&self) -> usize {
        use faer::Unbind;

        self.0.as_mat_ref().ncols().unbound()
    }

    #[inline]
    fn access(&self) -> matrixcompare::Access<'_, V> {
        matrixcompare::Access::Dense(self)
    }
}

#[cfg(feature = "compare")]
impl<T, V: Clone> matrixcompare::DenseAccess<V> for FaerCompare<T>
where
    T: faer::mat::AsMatRef<T = V>,
{
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> V {
        use faer::Shape;

        let mat = self.0.as_mat_ref();
        let row = unsafe { mat.nrows().unchecked_idx(row) };
        let col = unsafe { mat.ncols().unchecked_idx(col) };
        mat[(row, col)].clone()
    }
}

#[macro_export]
macro_rules! v {
    [$($components:expr $(,)?)*] => {
        $crate::dense::Vec::from_array([[$($components, )*]])
    };
}

#[macro_export]
macro_rules! m {
    [$($components:expr $(,)?)*] => {
        $crate::dense::Mat::from_array([$($components, )*])
    };
}
