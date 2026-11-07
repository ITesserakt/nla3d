use core::fmt::Debug;
use std::ops::Deref;

use faer::{
    Idx, Shape, ShapeIdx, sparse::{CreationError, FaerError, Pair, SparseColMatRef, SymbolicSparseColMat},
    traits::{Index, Symbolic},
};
use platform::collections::{vec, vec::Vec};
use platform::compat::boxed::Box;
use platform::compat::sync::Arc;

struct SparsityInfo<I = usize, Rows = usize, Cols = usize> {
    descriptor: Arc<SymbolicSparseColMat<I, Rows, Cols>>,
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct Visit<T>(T);

// FIXME: make sure that this struct is exact copy from faer
#[derive(Debug)]
struct Argsort<I> {
    idx: Vec<Visit<I>>,
    all_nnz: usize,
    nnz: usize,
}

impl<I: Index> Argsort<I> {
    fn apply_ordering<T>(&mut self, values: &mut [T])
    where
        I: Index,
    {
        // assert that no duplicate indicies found
        assert_eq!(self.all_nnz, self.nnz);
        assert_eq!(self.nnz, values.len());

        for visit in &mut self.idx {
            visit.set_visited(false);
        }

        for pos in 0..self.nnz {
            if self.idx[pos].is_visited() {
                continue;
            }

            let mut current = pos;
            loop {
                let visit = &mut self.idx[current];
                visit.set_visited(true);

                current = visit.get().zx();
                values.swap(pos, current);

                if pos == current {
                    break;
                }
            }
        }
    }
}

impl<I> From<faer::sparse::Argsort<I>> for Argsort<I> {
    #[inline]
    fn from(value: faer::sparse::Argsort<I>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
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

impl<T, I, Rows, Cols> Deref for Mat<T, I, Rows, Cols> {
    type Target = SymbolicSparseColMat<I, Rows, Cols>;

    fn deref(&self) -> &Self::Target {
        &self.info.descriptor
    }
}

impl<I: Index> Visit<I> {
    #[inline]
    fn is_visited(&self) -> bool {
        let top_bit = I::truncate(1 << (I::BITS - 1));
        self.0 & top_bit != I::truncate(0)
    }

    #[inline]
    fn set_visited(&mut self, visited: bool) {
        let top_bit = I::truncate(1 << (I::BITS - 1));
        let slot = &mut self.0;
        if visited {
            *slot = *slot | top_bit;
        } else {
            *slot = *slot & !top_bit
        }
    }

    #[inline]
    fn get(&self) -> I {
        let top_bit = I::truncate(1 << (I::BITS - 1));
        self.0 & !top_bit
    }
}

impl<I: Index, R: Shape, C: Shape> SparsityInfo<I, R, C> {
    #[inline]
    pub fn view(&self) -> SparseColMatRef<'_, I, Symbolic, R, C> {
        let descriptor = self.descriptor.deref().as_ref();
        SparseColMatRef::new(descriptor, Symbolic::materialize(descriptor.compute_nnz()))
    }
}

impl<T, I, R, C> UncompressedMat<R, C, T, I>
where
    R: Shape,
    C: Shape,
    I: Index,
{
    #[inline]
    pub fn compress(self) -> Mat<T, I, R, C> {
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

    pub fn try_new(mut uncompressed: UncompressedMat<R, C, T, I>) -> Result<Self, FaerError>
    where
        R: Shape,
        C: Shape,
        I: Index,
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
        let mut ordering = Argsort::from(ordering);
        ordering.apply_ordering(&mut uncompressed.values);

        Ok(Self {
            info: SparsityInfo {
                descriptor: Arc::new(descriptor),
            },
            items: uncompressed.values.into_boxed_slice(),
        })
    }

    pub fn view(&self) -> faer::sparse::SparseColMatRef<'_, I, T, R, C>
    where
        R: Shape,
        C: Shape,
        I: Index,
    {
        faer::sparse::SparseColMatRef::new(self.info.descriptor.deref().as_ref(), &self.items)
    }

    pub fn view_mut(&mut self) -> faer::sparse::SparseColMatMut<'_, I, T, R, C>
    where
        R: Shape,
        C: Shape,
        I: Index,
    {
        faer::sparse::SparseColMatMut::new(self.info.descriptor.deref().as_ref(), &mut self.items)
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
            info: SparsityInfo {
                descriptor: Arc::new(descriptor),
            },
            items: values.into_boxed_slice(),
        }
    }
}

impl<I: Index, R: Shape, C: Shape> Debug for SparsityInfo<I, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (nrows, ncols) = self.descriptor.shape();
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
    use crate::sparse::{Argsort, Mat, Visit};

    #[test]
    fn test_argsort_identity() {
        let mut argsort = Argsort {
            idx: vec![Visit(0_u32), Visit(1), Visit(2), Visit(3), Visit(4)],
            all_nnz: 5,
            nnz: 5,
        };
        let mut array = [2, 3, 1, 6, 2];

        argsort.apply_ordering(&mut array);
        assert_eq!(array, [2, 3, 1, 6, 2]);
    }

    #[test]
    fn test_argsort_empty() {
        let mut argsort = Argsort::<usize> {
            idx: vec![],
            all_nnz: 0,
            nnz: 0,
        };
        let mut array: [(); 0] = [];

        argsort.apply_ordering(&mut array);
        assert_eq!(array, []);
    }

    #[test]
    fn test_argsort() {
        let mut argsort = Argsort {
            idx: vec![
                Visit(1usize),
                Visit(5),
                Visit(4),
                Visit(3),
                Visit(2),
                Visit(0),
            ],
            all_nnz: 6,
            nnz: 6,
        };
        let mut array = [2, 6, 5, 4, 3, 1];

        argsort.apply_ordering(&mut array);
        assert_eq!(array, [1, 2, 3, 4, 5, 6])
    }

    #[test]
    fn test_multiple_argsort() {
        let mut argsort = Argsort {
            idx: vec![
                Visit(1usize),
                Visit(5),
                Visit(4),
                Visit(3),
                Visit(2),
                Visit(0),
            ],
            all_nnz: 6,
            nnz: 6,
        };
        let mut array = [2, 6, 5, 4, 3, 1];

        argsort.apply_ordering(&mut array);
        assert_eq!(array, [1, 2, 3, 4, 5, 6]);
        argsort.apply_ordering(&mut array);
        assert_eq!(array, [6, 1, 5, 4, 3, 2]);
        argsort.apply_ordering(&mut array);
        assert_eq!(array, [2, 6, 3, 4, 5, 1]);
    }

    #[test]
    fn test_new_mat() {
        let mut m = Mat::new((5, 5));
        m.add_entry(0usize, 0, 5.0).unwrap();
        m.add_entry(2, 2, 6.0).unwrap();
        m.add_entry(1, 1, 7.0).unwrap();
        m.add_entry(3, 4, 8.0).unwrap();
        m.add_entry(4, 3, 9.0).unwrap();
        let m = m.compress();

        dbg!(&m);
        assert_eq!(m.view().get(0, 0), Some(&5.0));
        assert_eq!(m.view().get(3, 4), Some(&8.0));
        assert_eq!(m.view().get(1, 1), Some(&7.0));
        assert_eq!(m.view().get(2, 2), Some(&6.0));
        assert_eq!(m.view().get(4, 3), Some(&9.0));
    }
}
