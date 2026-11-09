use core::fmt::Debug;
use core::marker::PhantomData;
use std::collections::btree_set::BTreeSet;

use generic_array::typenum::Const;
use proptest::collection::btree_set;
use proptest::prelude::prop::strategy::NewTree;
use proptest::prelude::prop::test_runner::TestRunner;
use proptest::prelude::{Arbitrary, Strategy};
use proptest::strategy::ValueTree;

use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
use crate::sparse;
use crate::zip::{
    Domain, Dynamic, ElementwiseConstructible, Full, KnownShape, Pattern, Static, Triangular, Zip,
};

pub struct MatrixStrategy<M, S, D, E> {
    shape: S,
    domain: D,
    element: E,
    _phantom: PhantomData<fn() -> M>,
}

pub struct MatrixTree<M, S, D, V> {
    shape: S,
    domain: D,
    trees: Vec<V>,
    next: usize,
    last: Option<usize>,
    _phantom: PhantomData<fn() -> M>,
}

impl<M, S, D, E> MatrixStrategy<M, S, D, E> {
    pub const fn new(shape: S, domain: D, element: E) -> Self {
        Self {
            shape,
            domain,
            element,
            _phantom: PhantomData,
        }
    }
}

impl<M, S, D, E: Debug> Debug for MatrixStrategy<M, S, D, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MatrixStrategy")
            .field("element", &self.element)
            .field("matrix", &std::any::type_name::<M>())
            .finish_non_exhaustive()
    }
}

fn count_points<S: KnownShape, D: Domain>(shape: S, domain: D) -> usize {
    Zip::from_fn(domain, |_| ())
        .with_shape(shape)
        .fold(0, |acc, ()| acc + 1)
}

impl<M, S, D, E> Strategy for MatrixStrategy<M, S, D, E>
where
    S: KnownShape,
    D: Domain,
    E: Strategy,
    M: ElementwiseConstructible<S, D, Item = E::Value> + Debug,
{
    type Tree = MatrixTree<M, S, D, E::Tree>;
    type Value = M;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let len = count_points(self.shape, self.domain.clone());
        let trees = (0..len)
            .map(|_| self.element.new_tree(runner))
            .collect::<Result<_, _>>()?;
        Ok(MatrixTree {
            shape: self.shape,
            domain: self.domain.clone(),
            trees,
            next: 0,
            last: None,
            _phantom: PhantomData,
        })
    }
}

impl<M, S, D, V> ValueTree for MatrixTree<M, S, D, V>
where
    S: KnownShape,
    D: Domain,
    V: ValueTree,
    M: ElementwiseConstructible<S, D, Item = V::Value> + Debug,
{
    type Value = M;

    fn current(&self) -> Self::Value {
        let mut trees = self.trees.iter();
        Zip::from_fn(self.domain.clone(), move |_| {
            trees.next().unwrap().current()
        })
        .with_shape(self.shape)
        .collect_into()
    }

    fn simplify(&mut self) -> bool {
        while self.next < self.trees.len() {
            if self.trees[self.next].simplify() {
                self.last = Some(self.next);
                return true;
            }
            self.next += 1;
        }
        false
    }

    fn complicate(&mut self) -> bool {
        let Some(k) = self.last else { return false };
        self.next = k;
        if self.trees[k].complicate() {
            true
        } else {
            self.last = None;
            false
        }
    }
}

pub const fn mat<const N: usize, const M: usize, E: Strategy>(
    element: E,
) -> MatrixStrategy<Mat<N, M, E::Value>, Static<N, M>, Full, E> {
    MatrixStrategy::new(Static, Full, element)
}

pub const fn mat_sym<const N: usize, K: SymKind, E: Strategy>(
    element: E,
) -> MatrixStrategy<MatSym<N, E::Value, K>, Static<N, N>, Triangular<K>, E>
where
    Const<N>: MatSymLength,
{
    MatrixStrategy::new(Static, Triangular(PhantomData), element)
}

pub const fn dmat<E: Strategy>(
    rows: usize,
    cols: usize,
    element: E,
) -> MatrixStrategy<DMat<E::Value>, Dynamic, Full, E> {
    MatrixStrategy::new(Dynamic { rows, cols }, Full, element)
}

impl<const N: usize, const M: usize, T: Arbitrary> Arbitrary for Mat<N, M, T> {
    type Parameters = T::Parameters;
    type Strategy = MatrixStrategy<Self, Static<N, M>, Full, T::Strategy>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        mat(T::arbitrary_with(args))
    }
}

impl<const N: usize, T: Arbitrary, K: SymKind> Arbitrary for MatSym<N, T, K>
where
    Const<N>: MatSymLength,
{
    type Parameters = T::Parameters;
    type Strategy = MatrixStrategy<Self, Static<N, N>, Triangular<K>, T::Strategy>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        mat_sym(T::arbitrary_with(args))
    }
}

pub fn sparse_mat<E>(
    rows: usize,
    cols: usize,
    max_per_col: usize,
    element: E,
) -> impl Strategy<Value = sparse::Mat<E::Value>>
where
    E: Strategy + Clone,
{
    use proptest::collection::vec;

    let per_col = max_per_col.min(rows);
    vec(btree_set(0..rows, 0..=per_col), cols).prop_flat_map(move |pattern| {
        let nnz = pattern.iter().map(|it| it.len()).sum::<usize>();
        vec(element.clone(), nnz).prop_map(move |values| build_sparse(rows, cols, &pattern, values))
    })
}

fn build_sparse<T>(
    rows: usize,
    cols: usize,
    pattern: &[BTreeSet<usize>],
    values: Vec<T>,
) -> sparse::Mat<T> {
    let mut col_ptr = Vec::with_capacity(cols + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for column in pattern {
        row_idx.extend(column.iter().copied());
        col_ptr.push(row_idx.len());
    }
    let pattern = Pattern::from_parts(col_ptr.into_boxed_slice(), None, row_idx.into_boxed_slice());
    let mut values = values.into_iter();

    Zip::from_fn(pattern, |_| values.next().unwrap())
        .with_shape(Dynamic { rows, cols })
        .collect()
}
