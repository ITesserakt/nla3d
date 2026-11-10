use core::array;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use generic_array::typenum::Const;
use nla3d_math::dense::{DMat, Lower, Mat, MatSym, MatSymLength};
use nla3d_math::matmul::{MatMul, MatMulAssign, manual_matmul_inplace};
use nla3d_math::zip::{Dynamic, Full, Static, Triangular, Zip};
use num_traits::ConstZero;
use rand::distr::StandardUniform;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_mat<const N: usize, const M: usize>(seed: u64) -> Mat<N, M, f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Zip::from_fn(Full, |_| rng.sample(StandardUniform))
        .with_shape(Static)
        .collect()
}

fn random_dmat(rows: usize, cols: usize, seed: u64) -> DMat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Zip::from_fn(Full, |_| rng.sample(StandardUniform))
        .with_shape(Dynamic { rows, cols })
        .collect()
}

fn random_matsym<const N: usize>(seed: u64) -> MatSym<N, f64, Lower>
where
    Const<N>: MatSymLength,
{
    let mut rng = StdRng::seed_from_u64(seed);
    Zip::from_fn(Triangular::default(), |_| rng.sample(StandardUniform))
        .with_shape(Static)
        .collect()
}

const SEEDS: [u64; 5] = [42, 123, 7, 9999, 314159];

fn bench_manual_square<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let mut a_mats = [Mat::<N, N, f64>::ZERO; SEEDS.len()];
    let mut b_mats = [Mat::<N, N, f64>::ZERO; SEEDS.len()];
    for (i, &seed) in SEEDS.iter().enumerate() {
        a_mats[i] = random_mat::<N, N>(seed);
        b_mats[i] = random_mat::<N, N>(seed.wrapping_add(1));
    }
    g.bench_function(BenchmarkId::new("manual_matmul_square", N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            for i in 0..SEEDS.len() {
                manual_matmul_inplace(N, &mut result, black_box(1.0_f64), &a_mats[i], &b_mats[i]);
            }
            black_box(result)
        });
    });
}

fn bench_faer_square<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let mut a_mats = [Mat::<N, N, f64>::ZERO; SEEDS.len()];
    let mut b_mats = [Mat::<N, N, f64>::ZERO; SEEDS.len()];
    for (i, &seed) in SEEDS.iter().enumerate() {
        a_mats[i] = random_mat::<N, N>(seed);
        b_mats[i] = random_mat::<N, N>(seed.wrapping_add(1));
    }
    g.bench_function(BenchmarkId::new("faer_matmul_square", N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            for i in 0..SEEDS.len() {
                matmul(
                    result.view_mut(),
                    Accum::Replace,
                    a_mats[i].view(),
                    b_mats[i].view(),
                    black_box(1.0_f64),
                    Par::Seq,
                );
            }
            black_box(result)
        });
    });
}

fn bench_trait_dispatch_square<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>) {
    let mut a_mats = [Mat::<N, N, f64>::ZERO; SEEDS.len()];
    let mut b_mats = [Mat::<N, N, f64>::ZERO; SEEDS.len()];
    for (i, &seed) in SEEDS.iter().enumerate() {
        a_mats[i] = random_mat::<N, N>(seed);
        b_mats[i] = random_mat::<N, N>(seed.wrapping_add(1));
    }
    g.bench_function(BenchmarkId::new("trait_matmul_square", N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            for i in 0..SEEDS.len() {
                result = black_box(&a_mats[i]).matmul(black_box(&b_mats[i]));
            }
            black_box(result)
        });
    });
}

fn manual_vs_faer_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_vs_faer_square");

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

fn bench_manual_rectangular<const N: usize, const M: usize, const L: usize>(
    g: &mut BenchmarkGroup<'_, WallTime>,
) {
    let mut a_mats = [Mat::<N, M, f64>::ZERO; SEEDS.len()];
    let mut b_mats = [Mat::<M, L, f64>::ZERO; SEEDS.len()];
    for (i, &seed) in SEEDS.iter().enumerate() {
        a_mats[i] = random_mat::<N, M>(seed);
        b_mats[i] = random_mat::<M, L>(seed.wrapping_add(1));
    }
    g.bench_function(
        BenchmarkId::new("manual_matmul_rect", format!("{N}x{M}x{L}")),
        |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<N, L, f64>::ZERO;
                for i in 0..SEEDS.len() {
                    manual_matmul_inplace(
                        M,
                        &mut result,
                        black_box(1.0_f64),
                        &a_mats[i],
                        &b_mats[i],
                    );
                }
                black_box(result)
            });
        },
    );
}

fn bench_faer_rectangular<const N: usize, const M: usize, const L: usize>(
    g: &mut BenchmarkGroup<'_, WallTime>,
) {
    let mut a_mats = [Mat::<N, M, f64>::ZERO; SEEDS.len()];
    let mut b_mats = [Mat::<M, L, f64>::ZERO; SEEDS.len()];
    for (i, &seed) in SEEDS.iter().enumerate() {
        a_mats[i] = random_mat::<N, M>(seed);
        b_mats[i] = random_mat::<M, L>(seed.wrapping_add(1));
    }
    g.bench_function(
        BenchmarkId::new("faer_matmul_rect", format!("{N}x{M}x{L}")),
        |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<N, L, f64>::ZERO;
                for i in 0..SEEDS.len() {
                    matmul(
                        result.view_mut(),
                        Accum::Replace,
                        a_mats[i].view(),
                        b_mats[i].view(),
                        black_box(1.0_f64),
                        Par::Seq,
                    );
                }
                black_box(result)
            });
        },
    );
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
    Const<N>: MatSymLength,
{
    let a_mats: [MatSym<N, f64, Lower>; SEEDS.len()] =
        array::from_fn(|i| random_matsym::<N>(SEEDS[i]));
    let b_mats: [MatSym<N, f64, Lower>; SEEDS.len()] =
        array::from_fn(|i| random_matsym::<N>(SEEDS[i].wrapping_add(1)));
    g.bench_function(BenchmarkId::new("matsym_manual", N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            for i in 0..SEEDS.len() {
                manual_matmul_inplace(N, &mut result, black_box(1.0_f64), &a_mats[i], &b_mats[i]);
            }
            black_box(result)
        });
    });
}

fn bench_matsym_faer<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>)
where
    Const<N>: MatSymLength,
{
    let a_mats: [MatSym<N, f64, Lower>; SEEDS.len()] =
        array::from_fn(|i| random_matsym::<N>(SEEDS[i]));
    let b_mats: [MatSym<N, f64, Lower>; SEEDS.len()] =
        array::from_fn(|i| random_matsym::<N>(SEEDS[i].wrapping_add(1)));
    let a_denses: [DMat; SEEDS.len()] = array::from_fn(|i| a_mats[i].to_dyn());
    let b_denses: [DMat; SEEDS.len()] = array::from_fn(|i| b_mats[i].to_dyn());
    g.bench_function(BenchmarkId::new("matsym_faer", N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            for i in 0..SEEDS.len() {
                matmul(
                    result.view_mut(),
                    Accum::Replace,
                    &a_denses[i],
                    &b_denses[i],
                    black_box(1.0_f64),
                    Par::Seq,
                );
            }
            black_box(result)
        });
    });
}

fn bench_matsym_trait<const N: usize>(g: &mut BenchmarkGroup<'_, WallTime>)
where
    Const<N>: MatSymLength,
{
    let a_mats: [MatSym<N, f64, Lower>; SEEDS.len()] =
        array::from_fn(|i| random_matsym::<N>(SEEDS[i]));
    let b_mats: [MatSym<N, f64, Lower>; SEEDS.len()] =
        array::from_fn(|i| random_matsym::<N>(SEEDS[i].wrapping_add(1)));
    g.bench_function(BenchmarkId::new("matsym_trait", N), |bencher| {
        bencher.iter(|| {
            let mut result = Mat::<N, N, f64>::ZERO;
            for i in 0..SEEDS.len() {
                result = black_box(&a_mats[i]).matmul(black_box(&b_mats[i]));
            }
            black_box(result)
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

    for n in [8, 16, 32, 64] {
        let a_mats: [DMat<f64>; SEEDS.len()] = array::from_fn(|i| random_dmat(n, n, SEEDS[i]));
        let b_mats: [DMat<f64>; SEEDS.len()] =
            array::from_fn(|i| random_dmat(n, n, SEEDS[i].wrapping_add(1)));
        group.bench_function(BenchmarkId::new("dmat_faer", n), |bencher| {
            let mut result = DMat::full(n, n, 0.0);
            bencher.iter(|| {
                for i in 0..SEEDS.len() {
                    matmul(
                        result.as_mut(),
                        Accum::Replace,
                        &a_mats[i],
                        &b_mats[i],
                        black_box(1.0_f64),
                        Par::Seq,
                    );
                }
            });
        });
    }

    group.finish();
}

fn matmul_assign_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_assign");
    group.sample_size(100);

    {
        let a_mats: [Mat<3, 3, f64>; SEEDS.len()] =
            array::from_fn(|i| random_mat::<3, 3>(SEEDS[i]));
        let b_mats: [Mat<3, 3, f64>; SEEDS.len()] =
            array::from_fn(|i| random_mat::<3, 3>(SEEDS[i].wrapping_add(1)));
        group.bench_function(BenchmarkId::new("matmul_assign", 3), |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<3, 3, f64>::ZERO;
                for i in 0..SEEDS.len() {
                    result.matmul_assign(black_box(&a_mats[i]), black_box(&b_mats[i]));
                }
                black_box(result)
            });
        });
    }

    {
        let a_mats: [Mat<16, 16, f64>; SEEDS.len()] =
            array::from_fn(|i| random_mat::<16, 16>(SEEDS[i]));
        let b_mats: [Mat<16, 16, f64>; SEEDS.len()] =
            array::from_fn(|i| random_mat::<16, 16>(SEEDS[i].wrapping_add(1)));
        group.bench_function(BenchmarkId::new("matmul_assign", 16), |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<16, 16, f64>::ZERO;
                for i in 0..SEEDS.len() {
                    result.matmul_assign(black_box(&a_mats[i]), black_box(&b_mats[i]));
                }
                black_box(result)
            });
        });
    }

    {
        let a_mats: [Mat<32, 32, f64>; SEEDS.len()] =
            array::from_fn(|i| random_mat::<32, 32>(SEEDS[i]));
        let b_mats: [Mat<32, 32, f64>; SEEDS.len()] =
            array::from_fn(|i| random_mat::<32, 32>(SEEDS[i].wrapping_add(1)));
        group.bench_function(BenchmarkId::new("matmul_assign", 32), |bencher| {
            bencher.iter(|| {
                let mut result = Mat::<32, 32, f64>::ZERO;
                for i in 0..SEEDS.len() {
                    result.matmul_assign(black_box(&a_mats[i]), black_box(&b_mats[i]));
                }
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
