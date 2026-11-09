use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, measurement::WallTime};
use std::hint::black_box;
use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use nla3d_math::dense::{DMat, Lower, Mat, MatSym};
use nla3d_math::matmul::{MatMul, MatMulAssign, manual_matmul_inplace};
use nla3d_math::zip::{Dynamic, Fill, RandomConstructible, Static, Zip};
use num_traits::ConstZero;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn random_mat<const N: usize, const M: usize>(seed: u64) -> Mat<N, M, f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    Mat::random_uniform(Static, &mut rng)
}

fn random_dmat(rows: usize, cols: usize, seed: u64) -> DMat<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    DMat::random_uniform(Dynamic { rows, cols }, &mut rng)
}

fn random_matsym<const N: usize>(seed: u64) -> MatSym<N, f64, Lower>
where
    generic_array::typenum::Const<N>: nla3d_math::dense::MatSymLength,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    MatSym::random_uniform(Static, &mut rng)
}

fn bench_manual_square<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let a = random_mat::<N, N>(42);
    let b_mat = random_mat::<N, N>(123);

    g.bench_function(&format!("manual_matmul_square_{}x{}", N, N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            manual_matmul_inplace(N, &mut result, black_box(1.0_f64), &a, &b_mat);
            black_box(result)
        });
    });
}

fn bench_faer_square<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let a = random_mat::<N, N>(42);
    let b_mat = random_mat::<N, N>(123);

    g.bench_function(&format!("faer_matmul_square_{}x{}", N, N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            matmul(result.view_mut(), Accum::Replace, a.view(), b_mat.view(), black_box(1.0_f64), Par::Seq);
            black_box(result)
        });
    });
}

fn bench_trait_dispatch_square<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let a = random_mat::<N, N>(42);
    let b_mat = random_mat::<N, N>(123);

    g.bench_function(&format!("trait_matmul_square_{}x{}", N, N), |bencher| {
        bencher.iter(|| {
            black_box(&a).matmul(black_box(&b_mat))
        });
    });
}

fn manual_vs_faer_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_vs_faer_square");
    group.sample_size(100);

    bench_manual_square::<1>(&mut group);
    bench_faer_square::<1>(&mut group);
    bench_trait_dispatch_square::<1>(&mut group);

    bench_manual_square::<2>(&mut group);
    bench_faer_square::<2>(&mut group);
    bench_trait_dispatch_square::<2>(&mut group);

    bench_manual_square::<3>(&mut group);
    bench_faer_square::<3>(&mut group);
    bench_trait_dispatch_square::<3>(&mut group);

    bench_manual_square::<4>(&mut group);
    bench_faer_square::<4>(&mut group);
    bench_trait_dispatch_square::<4>(&mut group);

    bench_manual_square::<5>(&mut group);
    bench_faer_square::<5>(&mut group);
    bench_trait_dispatch_square::<5>(&mut group);

    bench_manual_square::<8>(&mut group);
    bench_faer_square::<8>(&mut group);
    bench_trait_dispatch_square::<8>(&mut group);

    bench_manual_square::<16>(&mut group);
    bench_faer_square::<16>(&mut group);
    bench_trait_dispatch_square::<16>(&mut group);

    bench_manual_square::<32>(&mut group);
    bench_faer_square::<32>(&mut group);
    bench_trait_dispatch_square::<32>(&mut group);

    group.finish();
}

fn bench_manual_rectangular<const N: usize, const M: usize, const L: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let a = random_mat::<N, M>(42);
    let b_mat = random_mat::<M, L>(123);

    g.bench_function(&format!("manual_matmul_rect_{}x{}x{}x{}", N, M, M, L), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, L, f64>::ZERO;
            manual_matmul_inplace(M, &mut result, black_box(1.0_f64), &a, &b_mat);
            black_box(result)
        });
    });
}

fn bench_faer_rectangular<const N: usize, const M: usize, const L: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let a = random_mat::<N, M>(42);
    let b_mat = random_mat::<M, L>(123);

    g.bench_function(&format!("faer_matmul_rect_{}x{}x{}x{}", N, M, M, L), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, L, f64>::ZERO;
            matmul(result.view_mut(), Accum::Replace, a.view(), b_mat.view(), black_box(1.0_f64), Par::Seq);
            black_box(result)
        });
    });
}

fn manual_vs_faer_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_vs_faer_rectangular");
    group.sample_size(100);

    bench_manual_rectangular::<2, 3, 2>(&mut group);
    bench_faer_rectangular::<2, 3, 2>(&mut group);

    bench_manual_rectangular::<3, 2, 3>(&mut group);
    bench_faer_rectangular::<3, 2, 3>(&mut group);

    bench_manual_rectangular::<4, 8, 4>(&mut group);
    bench_faer_rectangular::<4, 8, 4>(&mut group);

    bench_manual_rectangular::<8, 4, 8>(&mut group);
    bench_faer_rectangular::<8, 4, 8>(&mut group);

    bench_manual_rectangular::<10, 20, 10>(&mut group);
    bench_faer_rectangular::<10, 20, 10>(&mut group);

    bench_manual_rectangular::<20, 10, 20>(&mut group);
    bench_faer_rectangular::<20, 10, 20>(&mut group);

    group.finish();
}

fn bench_matsym_manual<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>)
where
    generic_array::typenum::Const<N>: nla3d_math::dense::MatSymLength,
{
    let a = random_matsym::<N>(42);
    let b_mat = random_matsym::<N>(123);

    g.bench_function(&format!("matsym_manual_{}x{}", N, N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            manual_matmul_inplace(N, &mut result, black_box(1.0_f64), &a, &b_mat);
            black_box(result)
        });
    });
}

fn bench_matsym_faer<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>)
where
    generic_array::typenum::Const<N>: nla3d_math::dense::MatSymLength,
{
    let a = random_matsym::<N>(42);
    let b_mat = random_matsym::<N>(123);
    let a_dense = a.to_mat();
    let b_dense = b_mat.to_mat();

    g.bench_function(&format!("matsym_faer_{}x{}", N, N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            matmul(result.view_mut(), Accum::Replace, a_dense.view(), b_dense.view(), black_box(1.0_f64), Par::Seq);
            black_box(result)
        });
    });
}

fn bench_matsym_trait<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>)
where
    generic_array::typenum::Const<N>: nla3d_math::dense::MatSymLength,
{
    let a = random_matsym::<N>(42);
    let b_mat = random_matsym::<N>(123);

    g.bench_function(&format!("matsym_trait_{}x{}", N, N), |bencher| {
        bencher.iter(|| {
            black_box(&a).matmul(black_box(&b_mat))
        });
    });
}

fn matsym_manual_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("matsym_manual_vs_faer");
    group.sample_size(100);

    bench_matsym_manual::<1>(&mut group);
    bench_matsym_faer::<1>(&mut group);
    bench_matsym_trait::<1>(&mut group);

    bench_matsym_manual::<2>(&mut group);
    bench_matsym_faer::<2>(&mut group);
    bench_matsym_trait::<2>(&mut group);

    bench_matsym_manual::<3>(&mut group);
    bench_matsym_faer::<3>(&mut group);
    bench_matsym_trait::<3>(&mut group);

    bench_matsym_manual::<4>(&mut group);
    bench_matsym_faer::<4>(&mut group);
    bench_matsym_trait::<4>(&mut group);

    bench_matsym_manual::<5>(&mut group);
    bench_matsym_faer::<5>(&mut group);
    bench_matsym_trait::<5>(&mut group);

    bench_matsym_manual::<8>(&mut group);
    bench_matsym_faer::<8>(&mut group);
    bench_matsym_trait::<8>(&mut group);

    group.finish();
}

fn dmat_matmul_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("dmat_matmul_baseline");
    group.sample_size(100);

    for &n in &[8, 16, 32, 64] {
        let a = random_dmat(n, n, 42);
        let b_mat = random_dmat(n, n, 123);

        group.bench_function(&format!("dmat_faer_{}x{}", n, n), |bencher| {
            bencher.iter(|| {
                let mut result: DMat<f64> = Zip::from(Fill(f64::ZERO))
                    .with_shape(Dynamic { rows: n, cols: n })
                    .collect_into();
                matmul(result.as_mut(), Accum::Replace, a.as_ref(), b_mat.as_ref(), black_box(1.0_f64), Par::Seq);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn matmul_assign_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_assign");
    group.sample_size(100);

    {
        let a = random_mat::<3, 3>(42);
        let b_mat = random_mat::<3, 3>(123);
        group.bench_function("matmul_assign_3x3", |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<3, 3, f64>::ZERO;
                result.matmul_assign(black_box(&a), black_box(&b_mat));
                black_box(result)
            });
        });
    }

    {
        let a = random_mat::<16, 16>(42);
        let b_mat = random_mat::<16, 16>(123);
        group.bench_function("matmul_assign_16x16", |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<16, 16, f64>::ZERO;
                result.matmul_assign(black_box(&a), black_box(&b_mat));
                black_box(result)
            });
        });
    }

    {
        let a = random_mat::<32, 32>(42);
        let b_mat = random_mat::<32, 32>(123);
        group.bench_function("matmul_assign_32x32", |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<32, 32, f64>::ZERO;
                result.matmul_assign(black_box(&a), black_box(&b_mat));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    manual_vs_faer_square,
    manual_vs_faer_rectangular,
    matsym_manual_vs_faer,
    dmat_matmul_baseline,
    matmul_assign_bench,
);
criterion_main!(benches);
