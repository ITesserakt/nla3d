#![allow(semicolon_in_expressions_from_non_local_macros)]

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use matrixcompare::{prop_assert_matrix_eq, prop_assert_scalar_eq};
use num_traits::Zero;
use platform::collections::hash_set::HashSet;
use proptest::prelude::*;
use proptest::{prop_assert, prop_assert_eq, prop_assume, proptest};

use crate::arbitrary::{dmat, mat, mat_sym, sparse_mat};
use crate::dense::{Lower, Mat, MatSym, SymKind, Upper};
use crate::matmul::{MatMul, MatMulAssign};
use crate::tests::utils::{
    Tracked, clear_drops, get_drops_sorted, make_symmetric, normal_f64, sym_case, visit,
};
use crate::zip::ext::{ConsumeExt, ExpandExt, Fill, FillStatic, OwnedExt, TransposeExt, TriExt};
use crate::zip::{Dynamic, Full, Meet, Triangular, Zip};
use crate::{FaerCompare, m, sparse};

mod utils;

proptest! {
    #![proptest_config(proptest::test_runner::Config {
        #[cfg(miri)]
        failure_persistence: None,
        #[cfg(miri)]
        cases: 32,
        ..Default::default()
    })]

    #[test]
    fn zip_map_roundtrip_mat(a in mat::<3, 4, _>(normal_f64())) {
        let result = Zip::from(&a).map(|x| *x);
        prop_assert_eq!(result, a);
    }

    #[test]
    fn zip_map_roundtrip_dmat(a in dmat(5, 7, normal_f64())) {
        let result = Zip::from(&a).map(|x| *x);
        prop_assert_matrix_eq!(FaerCompare(a), FaerCompare(result), comp=float);
    }

    #[test]
    fn zip_map_roundtrip_matsym_lower(s in mat_sym::<4, Lower, _>(normal_f64())) {
        let result = Zip::from(&s).map(|x| *x);
        prop_assert_eq!(result, s);
    }

    #[test]
    fn zip_map_roundtrip_matsym_upper(s in mat_sym::<3, Upper, _>(normal_f64())) {
        let result = Zip::from(&s).map(|x| *x);
        prop_assert_eq!(result, s);
    }

    #[test]
    fn zip_fold_sum_mat(a in mat::<3, 4, _>(normal_f64())) {
        let sum = Zip::from(&a).fold(0.0_f64, |acc, x| acc + *x);
        let manual: f64 = (0..3).flat_map(|i| (0..4).map(move |j| a[(i, j)])).sum();
        prop_assert_scalar_eq!(sum, manual, comp=float);
    }

    #[test]
    fn zip_fold_dot_product(a in mat::<5, 5, _>(normal_f64()), b in mat::<5, 5, _>(normal_f64())) {
        let dot = Zip::from(&a).and(&b).fold(0.0_f64, |acc, x, y| acc + *x * *y);
        let manual: f64 = (0..5).flat_map(|i| (0..5).map(move |j| a[(i, j)] * b[(i, j)])).sum();
        prop_assert_scalar_eq!(dot, manual, comp=float);
    }

    #[test]
    fn zip_and_composition(a in mat::<3, 3, _>(normal_f64()), b in mat::<3, 3, _>(normal_f64()),
                           c in mat::<3, 3, _>(normal_f64())) {
        let count_abc = Zip::from(&a).and(&b).and(&c).fold(0usize, |acc, _, _, _| acc + 1);
        let count_ab = Zip::from(&a).and(&b).fold(0usize, |acc, _, _| acc + 1);
        prop_assert_eq!(count_abc, count_ab);
        prop_assert_eq!(count_ab, 9);
    }

    #[test]
    fn from_fn_collect_roundtrip(rows in 1usize..10, cols in 1usize..10) {
        let expected = Zip::from_fn(Full, |ix| (*ix.i as f64) * 10.0 + *ix.j as f64)
            .with_shape(Dynamic { rows, cols })
            .collect();
        for i in 0..rows {
            for j in 0..cols {
                prop_assert_scalar_eq!(expected[(i, j)], i as f64 * 10.0 + j as f64, comp=float);
            }
        }
    }

    #[test]
    fn expand_ext_matsym_lower(s in mat_sym::<4, Lower, _>(normal_f64())) {
        let dense = s.to_mat();
        let mut matched = true;
        Zip::from(s.expand()).and(&dense).for_each(|sv, dv| {
            if *sv != *dv { matched = false; }
        });
        prop_assert!(matched);
    }

    #[test]
    fn tri_ext_lower_restricts(a in mat::<4, 4, _>(normal_f64())) {
        let count = Zip::from(a.lower()).fold(0usize, |acc, _| acc + 1);
        prop_assert_eq!(count, 4 * 5 / 2);
    }

    #[test]
    fn tri_ext_upper_restricts(a in mat::<4, 4, _>(normal_f64())) {
        let count = Zip::from(a.upper()).fold(0usize, |acc, _| acc + 1);
        prop_assert_eq!(count, 4 * 5 / 2);
    }

    #[test]
    fn consume_producer_sum(a in mat::<3, 3, _>(normal_f64())) {
        let sum = Zip::from(a.consume()).fold(0.0_f64, |acc, x| acc + x);
        let manual: f64 = (0..3).flat_map(|i| (0..3).map(move |j| a[(i, j)])).sum();
        prop_assert_scalar_eq!(sum, manual, comp=float);
    }

    #[test]
    fn to_owned_producer_mat(a in mat::<3, 3, _>(normal_f64())) {
        let owned = Zip::from(a.as_owned()).map(|x| x);
        prop_assert_eq!(owned, a);
    }

    #[test]
    fn fill_broadcast(rows in 1usize..10, cols in 1usize..10, value in normal_f64()) {
        let result = Zip::from(Fill(value))
            .with_shape(Dynamic { rows, cols })
            .collect();
        for i in 0..rows {
            for j in 0..cols {
                prop_assert_scalar_eq!(result[(i, j)], value, comp=float);
            }
        }
    }

    #[test]
    fn fill_static_broadcast(value in normal_f64()) {
        let result = Zip::from(FillStatic::<3, 4, _>(value)).collect();
        for i in 0..3 {
            for j in 0..4 {
                prop_assert_scalar_eq!(result[(i, j)], value, comp=float);
            }
        }
    }

    #[test]
    fn zip_three_producers(a in mat::<3, 3, _>(normal_f64()), b in mat::<3, 3, _>(normal_f64()),
                           c in mat::<3, 3, _>(normal_f64())) {
        let result = Zip::from(&a).and(&b).and(&c).map(|x, y, z| *x + *y + *z);
        for i in 0..3 {
            for j in 0..3 {
                let expected = a[(i, j)] + b[(i, j)] + c[(i, j)];
                prop_assert_scalar_eq!(result[(i, j)], expected, comp=float);
            }
        }
    }

    #[test]
    fn domain_meet_triangular_lower(n in 1usize..10) {
        let d1 = Triangular::<Lower>::default();
        let d2 = Triangular::<Lower>::default();
        let result = d1.meet(d2);
        let v = visit(Dynamic { rows: n, cols: n }, result);
        prop_assert_eq!(v.len(), n * (n + 1) / 2);
    }

    #[test]
    fn add_commutativity(a in mat::<3, 4, _>(normal_f64()), b in mat::<3, 4, _>(normal_f64())) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn add_associativity(a in mat::<3, 3, _>(normal_f64()), b in mat::<3, 3, _>(normal_f64()),
                         c in mat::<3, 3, _>(normal_f64())) {
        prop_assert_matrix_eq!((a + b) + c, a + (b + c), comp=float);
    }

    #[test]
    fn add_identity(a in mat::<3, 4, _>(normal_f64())) {
        let zero = Mat::<3, 4, f64>::zero();
        prop_assert_eq!(a + zero, a);
    }

    #[test]
    fn mul_distributivity(a in mat::<3, 3, _>(normal_f64()), b in mat::<3, 3, _>(normal_f64()),
                           c in mat::<3, 3, _>(normal_f64())) {
        prop_assert_matrix_eq!(a * (b + c), a * b + a * c, comp=float);
    }

    #[test]
    fn scalar_broadcast_via_zip(a in mat::<3, 4, _>(normal_f64()), s in normal_f64()) {
        let zip_result = Zip::from(&a).and(Fill(s)).map(|x, y| *x * y);
        prop_assert_matrix_eq!(a * s, zip_result, comp=float);
    }

    #[test]
    fn matsym_from_mat_roundtrip(m in mat::<4, 4, _>(normal_f64())) {
        let sym = make_symmetric(m);
        let s = MatSym::<4, f64, Lower>::from_mat(sym);
        let back = s.to_mat();
        prop_assert_matrix_eq!(sym, back, comp=float);
    }

    #[test]
    fn matsym_to_mat_symmetric(s in mat_sym::<4, Lower, _>(normal_f64())) {
        let dense = s.to_mat();
        for i in 0..4 {
            for j in 0..4 {
                prop_assert_scalar_eq!(dense[(i, j)], dense[(j, i)], comp=float);
            }
        }
    }

    #[test]
    fn det_inv_product_3(m in mat::<3, 3, _>(normal_f64())) {
        prop_assume!(m.determinant().abs() > 1e-9);
        let product = m.matmul(m.inv());

        prop_assert_matrix_eq!(
            product,
            FaerCompare(crate::dense::DMat::identity(3, 3)),
            comp=abs, tol=1e-9
        );
    }

    #[test]
    fn det_inv_product_2(m in mat::<2, 2, _>(normal_f64())) {
        prop_assume!(m.determinant().abs() > 1e-9);
        let product = m.matmul(m.inv());

        prop_assert_matrix_eq!(
            product,
            FaerCompare(crate::dense::DMat::identity(2, 2)),
            comp=abs, tol=1e-9
        );
    }

    #[test]
    fn into_dyn_fidelity(a in mat::<3, 4, _>(normal_f64())) {
        let d = a.into_dyn();
        prop_assert_eq!(d.nrows(), 3);
        prop_assert_eq!(d.ncols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                prop_assert_scalar_eq!(d[(i, j)], a[(i, j)], comp=float);
            }
        }
    }

    #[test]
    fn accessor_3x3_consistency(a in mat::<3, 3, _>(normal_f64())) {
        prop_assert_eq!(a.xx, a[(0, 0)]);
        prop_assert_eq!(a.xy, a[(0, 1)]);
        prop_assert_eq!(a.xz, a[(0, 2)]);
        prop_assert_eq!(a.yx, a[(1, 0)]);
        prop_assert_eq!(a.yy, a[(1, 1)]);
        prop_assert_eq!(a.yz, a[(1, 2)]);
        prop_assert_eq!(a.zx, a[(2, 0)]);
        prop_assert_eq!(a.zy, a[(2, 1)]);
        prop_assert_eq!(a.zz, a[(2, 2)]);
    }

    #[test]
    fn sparse_construction_roundtrip(s in sparse_mat(5, 5, 3, normal_f64())) {
        for i in 0..5 {
            for j in 0..5 {
                let view_val = s.view().get(i, j);
                let get_val = s.get(i, j);
                prop_assert_eq!(view_val.is_some(), get_val.is_some());
                if let (Some(v), Some(g)) = (view_val, get_val) {
                    prop_assert_eq!(v, g);
                }
            }
        }
    }

    #[test]
    fn sparse_pattern_consistency(s in sparse_mat(5, 5, 3, normal_f64())) {
        for i in 0..5 {
            for j in 0..5 {
                let sparsity_loc = s.sparsity().get(i, j);
                let view_val = s.view().get(i, j);
                prop_assert_eq!(sparsity_loc.is_some(), view_val.is_some());
            }
        }
    }

    #[test]
    fn compute_nnz_accuracy(s in sparse_mat(5, 5, 3, normal_f64())) {
        let mut actual_count = 0;
        for i in 0..5 {
            for j in 0..5 {
                if s.view().get(i, j).is_some() {
                    actual_count += 1;
                }
            }
        }
        prop_assert_eq!(s.compute_nnz(), actual_count);
    }

    #[test]
    fn sparsity_get_out_of_bounds(s in sparse_mat(5, 5, 3, normal_f64())) {
        prop_assert!(s.sparsity().get(5, 0).is_none());
        prop_assert!(s.sparsity().get(0, 5).is_none());
    }

    #[test]
    fn matmul_static_2x3_3x4(a in mat::<2, 3, _>(normal_f64()), b in mat::<3, 4, _>(normal_f64())) {
        let our_result = a.matmul(b);
        let faer_result = a.into_dyn() * b.into_dyn();
        prop_assert_eq!(our_result.shape(), (2, 4));
        for i in 0..2 {
            for j in 0..4 {
                prop_assert_scalar_eq!(our_result[(i, j)], faer_result[(i, j)], comp=float);
            }
        }
    }

    #[test]
    fn matmul_dynamic(a in dmat(5, 7, normal_f64()), b in dmat(7, 3, normal_f64())) {
        let our_result = a.clone().matmul(b.clone());
        let faer_result = a * b;
        prop_assert_eq!(our_result.nrows(), 5);
        prop_assert_eq!(our_result.ncols(), 3);
        prop_assert_matrix_eq!(FaerCompare(our_result), FaerCompare(faer_result), comp=float);
    }

    #[test]
    fn matmul_matsym_matsym(a in mat_sym::<4, Lower, _>(normal_f64()), b in mat_sym::<4, Lower, _>(normal_f64())) {
        let our_result = a.to_mat().matmul(b.to_mat());
        let faer_result = a.to_mat().into_dyn() * b.to_mat().into_dyn();
        prop_assert_matrix_eq!(FaerCompare(our_result.into_dyn()), FaerCompare(faer_result), comp=float);
    }

    #[test]
    fn matmul_assign_correct(a in mat::<3, 3, _>(normal_f64()), b in mat::<3, 3, _>(normal_f64())) {
        let expected = a.matmul(b.clone());
        let mut c = Mat::<3, 3, f64>::zero();
        c.matmul_assign(a, b);
        prop_assert_matrix_eq!(c, expected, comp=float);
    }

    #[test]
    fn matmul_associativity(a in mat::<3, 3, _>(normal_f64()), b in mat::<3, 3, _>(normal_f64()),
                            c in mat::<3, 3, _>(normal_f64())) {
        let left = a.matmul(b.clone()).matmul(c.clone());
        let right = a.matmul(b.matmul(c));
        prop_assert_matrix_eq!(left, right, comp=float);
    }

    #[test]
    fn matmul_identity_left(a in mat::<3, 3, _>(normal_f64())) {
        let identity = m!([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let result = identity.matmul(a.clone());
        prop_assert_matrix_eq!(result, a, comp=float);
    }

    #[test]
    fn matmul_identity_right(a in mat::<3, 3, _>(normal_f64())) {
        let identity = m!([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let result = a.matmul(identity);
        prop_assert_matrix_eq!(result, a, comp=float);
    }

    #[test]
    fn zip_add_equals_builtin(a in mat::<3, 4, _>(normal_f64()), b in mat::<3, 4, _>(normal_f64())) {
        let zip_result = Zip::from(&a).and(&b).map(|x, y| *x + *y);
        let builtin = a + b;
        prop_assert_eq!(zip_result, builtin);
    }

    #[test]
    fn zip_sub_equals_builtin(a in mat::<3, 4, _>(normal_f64()), b in mat::<3, 4, _>(normal_f64())) {
        let zip_result = Zip::from(&a).and(&b).map(|x, y| *x - *y);
        let builtin = a - b;
        prop_assert_eq!(zip_result, builtin);
    }

    #[test]
    fn matsym_expand_zip_dense(s in mat_sym::<4, Lower, _>(normal_f64()), d in mat::<4, 4, _>(normal_f64())) {
        let dense_s = s.to_mat();
        let result = Zip::from(s.expand()).and(&d).map(|sv, dv| {
            sv * dv
        });
        for i in 0..4 {
            for j in 0..4 {
                let expected = dense_s[(i, j)] * d[(i, j)];
                prop_assert_scalar_eq!(result[(i, j)], expected, comp=float);
            }
        }
    }

    #[test]
    fn consume_producer_matsym(s in mat_sym::<3, Lower, _>(normal_f64())) {
        let mut expected = 0.0_f64;
        for i in 0..3 {
            for j in 0..=i {
                expected += s[(i, j)];
            }
        }
        let sum = Zip::from(s.consume()).fold(0.0_f64, |acc, x| acc + x);
        prop_assert_scalar_eq!(sum, expected, comp=float);
    }

    #[test]
    fn sparse_matmul_vs_faer(s in sparse_mat(10, 5, 3, normal_f64()), d in dmat(5, 4, normal_f64())) {
        let faer_result = s.view() * d.as_ref();
        let our_result = s.matmul(d.clone());
        prop_assert_eq!(our_result.nrows(), faer_result.nrows());
        prop_assert_eq!(our_result.ncols(), faer_result.ncols());
        prop_assert_matrix_eq!(FaerCompare(our_result), FaerCompare(faer_result), comp=float);
    }

    #[test]
    fn sparse_dense_matmul_shape(a in sparse_mat(200, 10, 6, proptest::num::f64::NORMAL), b in dmat(10, 4, proptest::num::f64::NORMAL)) {
        let out = a.matmul(b);
        prop_assert_eq!(out.nrows(), 200);
        prop_assert_eq!(out.ncols(), 4);
    }

    #[test]
    fn sparse_mat_passes_faer_checks(s in sparse_mat(10, 10, 5, any::<f64>())) {
        for i in 0..10 {
            for j in 0..10 {
                let position = s.sparsity().get(i, j);
                let faer_result = s.view().get(i, j);
                prop_assert_eq!(position.is_some(), faer_result.is_some());
                if let Some(position) = position {
                    let value = unsafe { s.get_unchecked(position) };
                    prop_assert_eq!(value, faer_result.unwrap());
                }
            }
        }
    }

    #[test]
    fn convert_dmat_to_sparse(m in dmat(20, 20, any::<f64>()), threshold in any::<f64>()) {
        let mut uncompressed = sparse::Mat::new((20, 20));
        Zip::from_fn(Full, |ix| (*ix.i, *ix.j)).and(&m).for_each(|(i, j), value| {
            if value.abs() < threshold { return; }
            uncompressed.add_entry(i, j, *value).unwrap();
        });
        let mat = uncompressed.compress();
        prop_assert!(mat.compute_nnz() <= 400);
        let all_eq = Zip::from(&mat).and(&m).fold(true, |acc, s, d| {
            acc && s == d
        });
        prop_assert!(all_eq);
        let all_eq_expanded = Zip::from(mat.expand()).and(&m).fold(true, |acc, s, d| {
            match s {
                Some(s) => acc && s == d,
                None => acc && *d < threshold
            }
        });
        prop_assert!(all_eq_expanded);
    }

    #[test]
    fn matmul_static_scalar(a in mat::<2, 4, _>(normal_f64()), b in mat::<4, 3, _>(normal_f64()), s in normal_f64()) {
        let da = a.into_dyn();
        let db = b.into_dyn();
        let out_manual = a.matmul_scalar(b, s);
        let out_faer = s * da * db;
        prop_assert_eq!(out_manual.shape(), out_faer.shape());
        prop_assert_matrix_eq!(out_manual, FaerCompare(out_faer), comp=float);
    }

    #[test]
    fn mirror_is_consistent_lower(s in mat_sym::<4, Lower, _>(any::<f64>())) {
        prop_assert!(s == s.to_mat());
    }

    #[test]
    fn mirror_is_consistent_upper(s in mat_sym::<4, Upper, _>(any::<f64>())) {
        prop_assert!(s == s.to_mat());
    }

    #[test]
    fn dynamic_shape(m in dmat(5, 7, any::<f64>())) {
        prop_assert_eq!(m.nrows(), 5);
        prop_assert_eq!(m.ncols(), 7);
    }

    #[test]
    fn sparse_stays_in_pattern(m in sparse_mat(5, 4, 3, any::<f64>())) {
        let stored = (0..5)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .filter(|&(i, j)| m.view().get(i, j).is_some())
            .count();
        prop_assert_eq!(stored, m.view().compute_nnz());
        prop_assert!(stored <= 4 * 3);
    }

    #[test]
    fn full_domain(rows in 0usize..12, cols in 0usize..12) {
        let v = visit(Dynamic { rows, cols }, Full);
        prop_assert_eq!(v.len(), rows * cols);
        for (k, p) in v.into_iter().enumerate() {
            prop_assert_eq!(p, (k % rows, k / rows, k));
        }
    }

    #[test]
    fn lower_domain(n in 0usize..20) {
        let v = visit(Dynamic { rows: n, cols: n }, Triangular::<Lower>::default());
        prop_assert_eq!(v.len(), n * (n + 1) / 2);
        let unique: HashSet<_> = v.iter().map(|p| (p.0, p.1)).collect();
        prop_assert_eq!(unique.len(), v.len());
        for (k, (i, j, pos)) in v.into_iter().enumerate() {
            prop_assert!(j <= i && i < n);
            prop_assert_eq!(pos, k);
            prop_assert_eq!(Lower::flat_index(n, i, j), pos);
        }
    }

    #[test]
    fn upper_domain(n in 0usize..20) {
        let v = visit(Dynamic { rows: n, cols: n }, Triangular::<Upper>::default());
        prop_assert_eq!(v.len(), n * (n + 1) / 2);
        let unique: HashSet<_> = v.iter().map(|p| (p.0, p.1)).collect();
        prop_assert_eq!(unique.len(), v.len());
        for (k, (i, j, pos)) in v.into_iter().enumerate() {
            prop_assert!(i <= j && j < n);
            prop_assert_eq!(pos, k);
            prop_assert_eq!(Upper::flat_index(n, i, j), pos);
        }
    }

    #[test]
    fn upper_index((n, i, j) in sym_case()) {
        let k = Upper::flat_index(n, i, j);
        prop_assert!(k < n * (n + 1) / 2);
        prop_assert_eq!(k, Upper::flat_index(n, j, i));
        let (a, b) = Upper::wide_index(n, k);
        prop_assert!(a <= b && (a, b) == (i.min(j), i.max(j)));
        prop_assert_eq!(Upper::flat_index(n, a, b), k);
    }

    #[test]
    fn sym_updated_from_mat(m in mat::<4, 4, _>(any::<i32>())) {
        let mut s = MatSym::<4, i32, Lower>::zero();
        Zip::from(&mut s).and(&m).for_each(|x, y| *x = *y);
        prop_assert_eq!(s, MatSym::from_mat(m));
    }

    #[test]
    fn mat_touched_only_on_triangle(a in mat::<4, 4, _>(any::<i32>()), b in mat::<4, 4, _>(any::<i32>())) {
        let mut out = a;
        Zip::from(&mut out).and(b.lower()).for_each(|x, y| *x = *y);
        for i in 0..4 { for j in 0..4 {
            prop_assert_eq!(out[(i, j)], if i >= j { b[(i, j)] } else { a[(i, j)] });
        }}
    }

    #[test]
    fn consume_drops_each_once(panic_at in 0usize..=9, tri in any::<bool>()) {
        clear_drops();
        let m = Mat::<3, 3, _>::from_array(std::array::from_fn(|j| std::array::from_fn(|i| Tracked(j * 3 + i))));
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let mut seen = 0;
            let f = |x: Tracked| { if seen == panic_at { panic!() } seen += 1; drop(x) };
            if tri { Zip::from(m.consume().lower()).for_each(f) }
            else   { Zip::from(m.consume()).for_each(f) }
        }));
        let d = get_drops_sorted();
        prop_assert_eq!(d, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn add_variants_agree(a in mat::<3, 3, _>(any::<f64>()), b in mat::<3, 3, _>(any::<f64>())) {
        let r = a + b;
        prop_assert_eq!(&a + b, r);
        prop_assert_eq!(a + &b, r);
        prop_assert_eq!(&a + &b, r);
        let (mut c, mut d) = (a, a);
        c += b; d += &b;
        prop_assert_eq!(c, r);
        prop_assert_eq!(d, r);
        prop_assert_eq!(r, b + a);
    }

    #[test]
    fn inverse_involution3(m in mat::<3, 3, _>(normal_f64())) {
        let back = m.try_inv().and_then(|it| it.try_inv());
        prop_assume!(back.is_ok());
        prop_assert_matrix_eq!(m, back.unwrap(), comp=abs, tol=1.0);
    }

    #[test]
    fn inverse_involution2(m in mat::<2, 2, _>(normal_f64())) {
        let back = m.try_inv().and_then(|it| it.try_inv());
        prop_assume!(back.is_ok());
        prop_assert_matrix_eq!(m, back.unwrap(), comp=abs, tol=1e-6);
    }

    #[test]
    fn inverse_involution1(m in mat::<1, 1, _>(normal_f64())) {
        let back = m.try_inv().and_then(|it| it.try_inv());
        prop_assume!(back.is_ok());
        prop_assert_matrix_eq!(m, back.unwrap(), comp=float);
    }

    #[test]
    fn det3_sym_matches_faer(m in mat_sym::<3, Lower, _>(normal_f64())) {
        let d = m.determinant();
        let f = m.to_mat().determinant();
        prop_assert_scalar_eq!(d, f, comp=float);
    }

    #[test]
    fn det2_sym_matches_faer(m in mat_sym::<2, Lower, _>(normal_f64())) {
        let d = m.determinant();
        let f = m.to_mat().determinant();
        prop_assert_scalar_eq!(d, f, comp=float);
    }

    #[test]
    fn det1_sym_matches_faer(m in mat_sym::<1, Lower, _>(normal_f64())) {
        let d = m.determinant();
        let f = m.to_mat().determinant();
        prop_assert_scalar_eq!(d, f, comp=float);
    }

    #[test]
    fn inverse_involution3_sym(m in mat_sym::<3, Lower, _>(normal_f64())) {
        let back = m.try_inv().and_then(|it| it.try_inv());
        prop_assume!(back.is_ok());
        prop_assert_matrix_eq!(m, back.unwrap(), comp=abs, tol=1.0);
    }

    #[test]
    fn inverse_involution2_sym(m in mat_sym::<2, Lower, _>(normal_f64())) {
        let back = m.try_inv().and_then(|it| it.try_inv());
        prop_assume!(back.is_ok());
        prop_assert_matrix_eq!(m, back.unwrap(), comp=abs, tol=1e-6);
    }

    #[test]
    fn inverse_involution1_sym(m in mat_sym::<1, Lower, _>(normal_f64())) {
        let back = m.try_inv().and_then(|it| it.try_inv());
        prop_assume!(back.is_ok());
        prop_assert_matrix_eq!(m, back.unwrap(), comp=float);
    }

    #[test]
    fn double_transpose_on_smat(m in mat_sym::<5, Lower, _>(any::<f64>())) {
        let double_transposed = Zip::from(m.consume().transpose().transpose()).collect();
        prop_assert_matrix_eq!(m, double_transposed, comp=float);
    }

    #[test]
    fn double_transpose_intermediate_on_smat(m in mat_sym::<5, Lower, _>(any::<f64>())) {
        let transposed = m.consume().transposed();
        let double_transposed = transposed.consume().transposed();
        prop_assert_matrix_eq!(m, double_transposed, comp=float);
    }

    #[test]
    fn double_transpose_on_dmat(m in dmat(5, 3, any::<f64>())) {
        let double_transposed = Zip::from(m.as_owned().transpose().transpose()).collect();
        prop_assert_matrix_eq!(FaerCompare(m), FaerCompare(double_transposed), comp=float);
    }

    #[test]
    fn double_transpose_intermediate_on_dmat(m in dmat(5, 3, any::<f64>())) {
        let transposed = m.as_owned().transposed();
        let double_transposed = transposed.as_owned().transposed();
        prop_assert_matrix_eq!(FaerCompare(m), FaerCompare(double_transposed), comp=float);
    }

    #[test]
    fn self_transpose_matmul_is_symmetric_static(a in mat::<3, 5, _>(normal_f64())) {
        let aat = a.matmul(a.transpose());
        let ata = a.transpose().matmul(a);

        prop_assert_matrix_eq!(aat, aat.consume().transposed(), comp=float);
        prop_assert_matrix_eq!(ata, ata.consume().transposed(), comp=float);
    }
}

#[test]
fn sym_mat_size() {
    use std::mem::size_of;
    assert_eq!(size_of::<MatSym<3>>(), 6 * size_of::<f64>());
}

#[test]
fn sym_mat_new() {
    let mut mat = MatSym::<3>::zero();
    mat[(1, 0)] = 3.0;
    assert_eq!(mat[(1, 0)], 3.0);
    assert_eq!(mat[(0, 1)], 3.0);
}

#[test]
fn sym_mat_eq() {
    let mut mat = MatSym::<3>::zero();
    mat[(1, 0)] = 3.0;
    let mut mat2 = Mat::<3, 3>::zero();
    mat2[(1, 0)] = 3.0;
    mat2[(0, 1)] = 3.0;
    assert_eq!(mat, mat2);
}

#[test]
fn upper_indexing() {
    let n = 5;
    for i in 0..n {
        for j in 0..n {
            let k = Upper::flat_index(n, i, j);
            let (i_, j_) = Upper::wide_index(n, k);
            assert!(i == i_ && j == j_ || i == j_ && j == i_);
        }
    }
}

#[test]
fn lower_indexing() {
    let n = 5;
    for i in 0..n {
        for j in 0..n {
            let k = Lower::flat_index(n, i, j);
            let (i_, j_) = Lower::wide_index(n, k);
            assert!(i == i_ && j == j_ || i == j_ && j == i_);
        }
    }
}

#[test]
fn vec_addition() {
    use crate::v;
    let a = v![1, 2, 3];
    let b = v![4, 5, 6];
    let c = v![5, 7, 9];
    assert_eq!(c, a + b);
    assert_eq!(c, &a + b);
    assert_eq!(c, a + &b);
    assert_eq!(c, &a + &b);
}

#[test]
fn vec_assign_addition() {
    use crate::v;
    let a = v![1, 2, 3];
    let mut b = v![0, 0, 0];
    b += a;
    assert_eq!(b, a);
}

#[test]
fn into_dyn() {
    use matrixcompare::assert_matrix_eq;
    let a = m![[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let b = a.into_dyn();
    assert_matrix_eq!(a, FaerCompare(b));
}

#[test]
fn zip_fold() {
    let a = m![[1, 2, 3], [2, 3, 4], [5, 6, 7]];
    let b = MatSym::<3, f64>::from_mat(m![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let sum = Zip::from(&a)
        .and(b.expand())
        .fold(0.0, |acc, x, y| acc + (*x as f64) * *y);
    assert_eq!(sum, 4.0);
}

#[test]
fn zip_for_each() {
    let a = m![[1, 2, 3], [2, 3, 4], [5, 6, 7]];
    let b = MatSym::<3, f64>::from_mat(m![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut total_non_zero = 0;
    Zip::from(&a).and(b.expand()).for_each(|x, y| {
        if (*x as f64) * *y != 0.0 {
            total_non_zero += 1;
        }
    });
    assert_eq!(total_non_zero, 2);
}

#[test]
fn zip_map_different_types() {
    let a = m![[1, 2, 3], [2, 3, 4], [5, 6, 7]];
    let b = MatSym::<3, f64>::from_mat(m![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let out = Zip::from(&a).and(&b).map(|x, y| (*x as f64) * *y);
    assert_eq!(out, m![[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
}

#[test]
fn zip_map_same_type() {
    let b = MatSym::<3, f64>::from_mat(m![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let out = Zip::from(b.expand())
        .and(b.expand())
        .map(|x, y| (*x * *y) as u8);
    assert_eq!(out, m![[0u8, 1, 0], [1, 0, 0], [0, 0, 0]]);
}

#[test]
fn for_each_mut_items_can_be_retained() {
    let mut a = m![[1, 2], [3, 4]].into_dyn();
    let mut v = Vec::new();
    Zip::from(&mut a).for_each(|x| v.push(x));
    for x in v {
        *x += 1;
    }
}

#[test]
fn change_output_type() {
    let a = m![[1, 2], [3, 4]];
    let b = Zip::from(a.consume()).map(|x| x);
    assert_eq!(a, b);
}

#[test]
fn zip_panicing() {
    use core::mem::ManuallyDrop;
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    struct IncOnDrop(u32);
    impl Drop for IncOnDrop {
        fn drop(&mut self) {
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    COUNTER.store(0, std::sync::atomic::Ordering::SeqCst);
    let a = m![[IncOnDrop(1), IncOnDrop(2)], [IncOnDrop(3), IncOnDrop(4)]];
    let result = catch_unwind(|| {
        Zip::from(a.consume()).for_each(|x| {
            let x = ManuallyDrop::new(x);
            if x.0 == 3 {
                panic!("Test");
            }
        });
    });
    assert_eq!(COUNTER.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(result.is_err());
}

#[test]
fn matmul_simple() {
    let identity = MatSym::from_mat(m![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let mat = m![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    assert_eq!(
        identity.matmul(mat),
        m![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    );
}

#[test]
fn sparse_new_mat() {
    use crate::sparse::Mat;
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

#[test]
fn sparse_new_mat_with_zip() {
    use crate::sparse_pattern;
    let pattern = sparse_pattern!(5 x 5 (
        0usize, 0;
        2, 2;
        1, 1;
        3, 4;
        4, 3
    ));
    let mat = Zip::from_fn(pattern, |ix| ix.pos as f64 + 5.0)
        .with_shape(Dynamic { rows: 5, cols: 5 })
        .collect();
    let view = mat.view();
    assert_eq!(view.get(0, 0), Some(&5.0));
    assert_eq!(view.get(1, 1), Some(&6.0));
    assert_eq!(view.get(2, 2), Some(&7.0));
    assert_eq!(view.get(4, 3), Some(&8.0));
    assert_eq!(view.get(3, 4), Some(&9.0));
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
    fn accessor_m33(m in crate::arbitrary::mat::<3, 3, _>(utils::normal_f64())) {
        prop_assert_eq!(m[(0, 0)], m.xx);
        prop_assert_eq!(m[(0, 1)], m.xy);
        prop_assert_eq!(m[(0, 2)], m.xz);
        prop_assert_eq!(m[(1, 0)], m.yx);
        prop_assert_eq!(m[(1, 1)], m.yy);
        prop_assert_eq!(m[(1, 2)], m.yz);
        prop_assert_eq!(m[(2, 0)], m.zx);
        prop_assert_eq!(m[(2, 1)], m.zy);
        prop_assert_eq!(m[(2, 2)], m.zz);
    }

    #[test]
    fn accessor_m32(m in crate::arbitrary::mat::<3, 2, _>(utils::normal_f64())) {
        prop_assert_eq!(m[(0, 0)], m.xx);
        prop_assert_eq!(m[(0, 1)], m.xy);
        prop_assert_eq!(m[(1, 0)], m.yx);
        prop_assert_eq!(m[(1, 1)], m.yy);
        prop_assert_eq!(m[(2, 0)], m.zx);
        prop_assert_eq!(m[(2, 1)], m.zy);
    }

    #[test]
    fn accessor_m23(m in crate::arbitrary::mat::<2, 3, _>(utils::normal_f64())) {
        prop_assert_eq!(m[(0, 0)], m.xx);
        prop_assert_eq!(m[(0, 1)], m.xy);
        prop_assert_eq!(m[(0, 2)], m.xz);
        prop_assert_eq!(m[(1, 0)], m.yx);
        prop_assert_eq!(m[(1, 1)], m.yy);
        prop_assert_eq!(m[(1, 2)], m.yz);
    }

    #[test]
    fn accessor_m22(m in crate::arbitrary::mat::<2, 2, _>(utils::normal_f64())) {
        prop_assert_eq!(m[(0, 0)], m.xx);
        prop_assert_eq!(m[(0, 1)], m.xy);
        prop_assert_eq!(m[(1, 0)], m.yx);
        prop_assert_eq!(m[(1, 1)], m.yy);
    }

    #[test]
    fn accessor_ms1(mu in crate::arbitrary::mat_sym::<1, Upper, _>(utils::normal_f64()),
                     ml in crate::arbitrary::mat_sym::<1, Lower, _>(utils::normal_f64())) {
        prop_assert_eq!(mu[(0, 0)], mu.x);
        prop_assert_eq!(ml[(0, 0)], ml.x);
    }

    #[test]
    fn accessor_ms2(mu in crate::arbitrary::mat_sym::<2, Upper, _>(utils::normal_f64()),
                     ml in crate::arbitrary::mat_sym::<2, Lower, _>(utils::normal_f64())) {
        prop_assert_eq!(mu[(0, 0)], mu.xx);
        prop_assert_eq!(mu[(0, 1)], mu.xy);
        prop_assert_eq!(mu[(1, 0)], mu.xy);
        prop_assert_eq!(mu[(1, 1)], mu.yy);
        prop_assert_eq!(ml[(0, 0)], ml.xx);
        prop_assert_eq!(ml[(0, 1)], ml.xy);
        prop_assert_eq!(ml[(1, 0)], ml.xy);
        prop_assert_eq!(ml[(1, 1)], ml.yy);
    }

    #[test]
    fn accessor_ms3(mu in crate::arbitrary::mat_sym::<3, Upper, _>(utils::normal_f64()),
                     ml in crate::arbitrary::mat_sym::<3, Lower, _>(utils::normal_f64())) {
        prop_assert_eq!(mu[(0, 0)], mu.xx);
        prop_assert_eq!(mu[(0, 1)], mu.xy);
        prop_assert_eq!(mu[(0, 2)], mu.xz);
        prop_assert_eq!(mu[(1, 0)], mu.xy);
        prop_assert_eq!(mu[(1, 1)], mu.yy);
        prop_assert_eq!(mu[(1, 2)], mu.yz);
        prop_assert_eq!(mu[(2, 0)], mu.xz);
        prop_assert_eq!(mu[(2, 1)], mu.yz);
        prop_assert_eq!(mu[(2, 2)], mu.zz);
        prop_assert_eq!(ml[(0, 0)], ml.xx);
        prop_assert_eq!(ml[(0, 1)], ml.xy);
        prop_assert_eq!(ml[(0, 2)], ml.xz);
        prop_assert_eq!(ml[(1, 0)], ml.xy);
        prop_assert_eq!(ml[(1, 1)], ml.yy);
        prop_assert_eq!(ml[(1, 2)], ml.yz);
        prop_assert_eq!(ml[(2, 0)], ml.xz);
        prop_assert_eq!(ml[(2, 1)], ml.yz);
        prop_assert_eq!(ml[(2, 2)], ml.zz);
    }
}
