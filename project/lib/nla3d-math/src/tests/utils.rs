//! Shared test utilities for property-based and unit tests.
//!
//! This module provides helper functions, types, and strategies used across
//! all test modules in the crate.

use core::cell::RefCell;

use num_traits::Zero;
use proptest::prelude::{Just, Strategy};

use crate::dense::Mat;
use crate::zip::{Domain, Dynamic, Zip};

/// Collect all visited positions from a domain into a vector of `(row, col, pos)` tuples.
pub fn visit<D: Domain>(shape: Dynamic, d: D) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    Zip::from_fn(d, |ix| (*ix.i, *ix.j, ix.pos))
        .with_shape(shape)
        .for_each(|it| v.push(it));
    v
}

/// Strategy that generates `(n, i, j)` where `n` is 1..5000 and `i, j` are in `0..n`.
/// Useful for testing symmetric matrix indexing.
pub fn sym_case() -> impl Strategy<Value = (usize, usize, usize)> {
    (1usize..5000).prop_flat_map(|n| (Just(n), 0..n, 0..n))
}

/// A wrapper type that records its value when dropped.
/// Used to verify that `ConsumeProducer` drops all elements even on panic.
pub struct Tracked(pub usize);

impl Drop for Tracked {
    fn drop(&mut self) {
        DROPS.with(|d| d.borrow_mut().push(self.0));
    }
}

thread_local!(static DROPS: RefCell<Vec<usize>> = RefCell::new(vec![]));

/// Clear the drop tracker. Call before each test that uses `Tracked`.
pub fn clear_drops() {
    DROPS.with(|d| d.borrow_mut().clear());
}

/// Get a sorted copy of the drop tracker.
pub fn get_drops_sorted() -> Vec<usize> {
    let mut d = DROPS.with(|d| d.borrow().clone());
    d.sort();
    d
}

/// Proptest strategy for f64 values with reasonable magnitude.
/// Uses a bounded range to avoid overflow in matrix operations (matmul, determinant, etc.).
/// Values in [-100, 100] are safe for 3x3 matmul (max product sum ~3*100*100 = 30000)
/// and determinant calculations without losing precision.
/// Returns a Clone strategy so it works with `sparse_mat`.
pub fn normal_f64() -> impl Strategy<Value = f64> + Clone {
    // Generate i32 in a wide range and convert to f64 — guarantees finite, no NaN/Inf
    (-10000i32..10000).prop_map(|v| v as f64)
}

/// Generate a symmetric `Mat<N, N, T>` from an arbitrary matrix by forcing `m[i,j] == m[j,i]`.
pub fn make_symmetric<const N: usize, T: Clone + Zero>(m: Mat<N, N, T>) -> Mat<N, N, T> {
    let mut result = Mat::zero();
    for i in 0..N {
        for j in 0..=i {
            let val = m[(i, j)].clone();
            result[(i, j)] = val.clone();
            result[(j, i)] = val;
        }
    }
    result
}
