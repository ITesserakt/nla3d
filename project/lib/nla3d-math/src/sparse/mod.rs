use core::fmt::Debug;
use std::ops::Deref;

use faer::prelude::SparseColMat;
use faer::sparse::{
    CreationError, FaerError, Pair, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef,
};
use faer::traits::{ComplexField, Index, Symbolic};
use faer::{Idx, Shape, ShapeIdx};
use platform::collections::vec;
use platform::collections::vec::Vec;
use platform::compat::boxed::Box;

use crate::zip::Pattern;

pub struct SparsityInfo<I = usize, Rows = usize, Cols = usize> {
    pub rows: Rows,
    pub cols: Cols,
    pub pattern: Pattern<I>,
}

pub struct UncompressedMat<Rows: ShapeIdx, Cols: ShapeIdx, T, I: Index> {
    shape: (Rows, Cols),
    entries: Vec<Pair<Idx<Rows, I>, Idx<Cols, I>>>,
    values: Vec<T>,
}

pub struct Mat<T = f64, I = usize, Rows = usize, Cols = usize> {
    info: SparsityInfo<I, Rows, Cols>,
    items: Box<[T]>,
}

impl<I: Index, R: Shape, C: Shape> SparsityInfo<I, R, C> {
    #[inline]
    pub fn view(&self) -> SparseColMatRef<'_, I, Symbolic, R, C> {
        let descriptor = self.descriptor();
        SparseColMatRef::new(descriptor, Symbolic::materialize(descriptor.compute_nnz()))
    }

    #[inline]
    pub fn descriptor(&self) -> SymbolicSparseColMatRef<'_, I, R, C> {
        unsafe {
            SymbolicSparseColMatRef::new_unchecked(
                self.rows,
                self.cols,
                &self.pattern.col_ptr,
                self.pattern.col_nnz.as_deref(),
                &self.pattern.row_idx,
            )
        }
    }

    #[inline]
    pub fn from_pattern(shape: (R, C), pattern: Pattern<I>) -> Self {
        SparsityInfo {
            rows: shape.0,
            cols: shape.1,
            pattern,
        }
    }

    #[inline]
    fn from_descriptor(descriptor: SymbolicSparseColMat<I, R, C>) -> Self {
        let (rows, cols, col_ptr, col_nnz, row_idx) = descriptor.into_parts();

        Self {
            rows,
            cols,
            pattern: Pattern::from_parts(
                col_ptr.into_boxed_slice(),
                col_nnz.map(|it| it.into_boxed_slice()),
                row_idx.into_boxed_slice(),
            ),
        }
    }
}

impl<I, R: Clone, C: Clone> Clone for SparsityInfo<I, R, C> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            rows: self.rows.clone(),
            cols: self.cols.clone(),
            pattern: self.pattern.clone(),
        }
    }
}

impl<I: Index, R: Shape, C: Shape> PartialEq for SparsityInfo<I, R, C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.pattern == other.pattern
    }
}

impl<T, I, R, C> UncompressedMat<R, C, T, I>
where
    R: Shape,
    C: Shape,
    I: Index,
{
    #[inline]
    pub fn compress(self) -> Mat<T, I, R, C>
    where
        T: ComplexField,
    {
        Mat::try_new(self).unwrap()
    }

    pub fn add_entry(&mut self, i: Idx<R, I>, j: Idx<C, I>, value: T) -> Result<(), CreationError> {
        self.entries.push(Pair { row: i, col: j });
        self.values.push(value);
        Ok(())
    }
}

impl<T, I, R, C> Mat<T, I, R, C> {
    #[inline]
    pub fn new(shape: (R, C)) -> UncompressedMat<R, C, T, I>
    where
        I: Index,
        R: ShapeIdx,
        C: ShapeIdx,
    {
        UncompressedMat {
            shape,
            entries: vec![],
            values: vec![],
        }
    }

    #[inline]
    pub fn sparsity(&self) -> &SparsityInfo<I, R, C>
    where
        R: Clone,
        C: Clone,
    {
        &self.info
    }

    #[inline]
    pub fn nrows(&self) -> R
    where
        R: Clone,
    {
        self.info.rows.clone()
    }

    #[inline]
    pub fn ncols(&self) -> C
    where
        C: Clone,
    {
        self.info.cols.clone()
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, pos: usize) -> &T {
        unsafe { self.items.get_unchecked(pos) }
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, pos: usize) -> &mut T {
        unsafe { self.items.get_unchecked_mut(pos) }
    }

    #[inline]
    pub(crate) fn from_parts(info: SparsityInfo<I, R, C>, items: Box<[T]>) -> Self {
        Self { info, items }
    }

    pub fn try_new(mut uncompressed: UncompressedMat<R, C, T, I>) -> Result<Self, FaerError>
    where
        R: Shape,
        C: Shape,
        I: Index,
        T: ComplexField,
    {
        let (descriptor, ordering) = match SymbolicSparseColMat::try_new_from_indices(
            uncompressed.shape.0,
            uncompressed.shape.1,
            &uncompressed.entries,
        ) {
            Ok(x) => x,
            Err(CreationError::Generic(e)) => return Err(e),
            Err(CreationError::OutOfBounds { row, col }) => {
                unreachable!("All elements should be inbounds: {row} {col}")
            }
        };
        let matrix = SparseColMat::new_from_argsort(descriptor, &ordering, &uncompressed.values)?;
        let (descriptor, values) = matrix.into_parts();

        Ok(Self {
            info: SparsityInfo::from_descriptor(descriptor),
            items: values.into_boxed_slice(),
        })
    }

    pub fn view(&self) -> faer::sparse::SparseColMatRef<'_, I, T, R, C>
    where
        R: Shape,
        C: Shape,
        I: Index,
    {
        faer::sparse::SparseColMatRef::new(self.info.descriptor().as_ref(), &self.items)
    }

    pub fn view_mut(&mut self) -> faer::sparse::SparseColMatMut<'_, I, T, R, C>
    where
        R: Shape,
        C: Shape,
        I: Index,
    {
        faer::sparse::SparseColMatMut::new(self.info.descriptor().as_ref(), &mut self.items)
    }
}

impl<I, T, R, C> From<faer::sparse::SparseColMat<I, T, R, C>> for Mat<T, I, R, C>
where
    R: Shape,
    C: Shape,
    I: Index,
{
    #[inline]
    fn from(value: faer::sparse::SparseColMat<I, T, R, C>) -> Self {
        let (descriptor, values) = value.into_parts();
        Self {
            info: SparsityInfo::from_descriptor(descriptor),
            items: values.into_boxed_slice(),
        }
    }
}

impl<I: Index, R: Shape, C: Shape> Debug for SparsityInfo<I, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (nrows, ncols) = (self.rows, self.cols);
        writeln!(f, "  ┌ {:>width$} ┐", "", width = 2 * ncols.unbound() - 1)?;
        let view = self.view();

        for i in R::indices(R::start(), nrows.end()) {
            write!(f, "  │")?;
            for j in C::indices(C::start(), ncols.end()) {
                match view.get(i, j) {
                    Some(Symbolic) => write!(f, "⋅ "),
                    None => write!(f, "  "),
                }?;
            }
            writeln!(f, " │")?;
        }

        writeln!(f, "  └ {:>width$} ┘", "", width = 2 * ncols.unbound() - 1)?;

        Ok(())
    }
}

impl<T: Debug, I: Index, R: Shape, C: Shape> Debug for Mat<T, I, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mat {{\n  info:\n{:?}  items: {:?},\n}}",
            self.info, self.items
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::num::f64::{ANY, NORMAL};
    use proptest::prelude::any;
    use proptest::{prop_assert, prop_assert_eq, proptest};

    use crate::arbitrary::{dmat, sparse_mat};
    use crate::matmul::MatMul;
    use crate::sparse::Mat;

    #[test]
    fn test_new_mat() {
        let mut m = Mat::new((5, 5));
        m.add_entry(0usize, 0, 5.0).unwrap();
        m.add_entry(2, 2, 6.0).unwrap();
        m.add_entry(1, 1, 7.0).unwrap();
        m.add_entry(3, 4, 8.0).unwrap();
        m.add_entry(4, 3, 9.0).unwrap();
        let m = m.compress();

        assert_eq!(m.view().get(0, 0), Some(&5.0));
        assert_eq!(m.view().get(3, 4), Some(&8.0));
        assert_eq!(m.view().get(1, 1), Some(&7.0));
        assert_eq!(m.view().get(2, 2), Some(&6.0));
        assert_eq!(m.view().get(4, 3), Some(&9.0));
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            #[cfg(miri)]
            failure_persistence: None,
            #[cfg(miri)]
            cases: 32,
            ..Default::default()
        })]

        #[test]
        fn sparse_dense_matmul(a in sparse_mat(200, 10, 6, NORMAL), b in dmat(10, 4, NORMAL)) {
            let out = a.matmul(b);

            prop_assert_eq!(out.nrows(), 200);
            prop_assert_eq!(out.ncols(), 4);
        }
    }
}
