use core::{
    f64::consts::{FRAC_PI_2, PI},
    ops::{Add, Div, Mul, Sub},
};

use nla3d_math::{
    dense::{Mat, Vec, zip::Zip}, m, num_traits::{MulAddAssign, One, Zero}, v,
};

const fn cos_approx_newton(x: f64, max_iterations: usize) -> f64 {
    let (x, sign) = if x > FRAC_PI_2 {
        (PI - x, -1.0)
    } else {
        (x, 1.0)
    };
    let (mut term, mut sum) = (1.0, 1.0);
    let mut k = 1;
    while k < max_iterations {
        term *= -x * x / ((2 * k - 1) * (2 * k)) as f64;
        sum += term;
        k += 1;
    }

    sign * sum
}

#[derive(Debug, Clone, Copy)]
pub struct GaussQuadrature<const N: usize, T = f64> {
    weights: Vec<N, T>,
    xs: Vec<N, T>,
}

#[derive(Debug, Clone)]
pub struct TensorGaussQuadrature<const D: usize, const N: usize, T = f64> {
    weights: Vec<N, T>,
    xs: Mat<N, D, T>,
}

pub trait Quadrature<const DIM: usize, T> {
    fn integrate(&self, f: impl FnMut(Mat<1, DIM, T>) -> T) -> T;
}

impl<Q, T, const D: usize> Quadrature<D, T> for &Q
where
    Q: Quadrature<D, T>,
{
    #[inline]
    fn integrate(&self, f: impl FnMut(Mat<1, D, T>) -> T) -> T {
        Q::integrate(self, f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearMap<T, Q, const D: usize> {
    from: Vec<D, T>,
    to: Vec<D, T>,
    inner: Q,
}

impl<const N: usize> GaussQuadrature<N, f64> {
    pub const fn new(max_iterations: usize) -> Self {
        let mut xs = [0.0; N];
        let mut ws = [0.0; N];
        let n = N as f64;

        let mut i = 0;
        while i < (N + 1) / 2 {
            let mut z = cos_approx_newton(PI * (i as f64 + 0.75) / (n + 0.5), max_iterations);
            let mut dp = 1.0;

            let mut it = 0;
            while it < max_iterations {
                let mut p_prev = 1.0;
                let mut p = z;
                let mut k = 2;
                while k <= N {
                    let kf = k as f64;
                    let p_next = ((2.0 * kf - 1.0) * z * p - (kf - 1.0) * p_prev) / kf;
                    p_prev = p;
                    p = p_next;
                    k += 1;
                }

                dp = n * (z * p - p_prev) / (z * z - 1.0);

                let dz = p / dp;
                z -= dz;
                if dz.abs() < 1e-15 {
                    break;
                };
                it += 1;
            }

            let w = 2.0 / ((1.0 - z * z) * dp * dp);
            xs[i] = -z;
            xs[N - 1 - i] = z;
            ws[i] = w;
            ws[N - 1 - i] = w;
            i += 1;
        }

        Self {
            weights: m!(ws),
            xs: m!(xs),
        }
    }
}

impl<const D: usize, const N: usize> TensorGaussQuadrature<D, N, f64> {
    pub const fn new<const M: usize>(max_iterations: usize) -> Self {
        const { assert!(N == M.pow(D as u32)) }

        let GaussQuadrature { weights, xs } = GaussQuadrature::<M>::new(max_iterations);

        let mut pts = [[0.0; N]; D];
        let mut ws = [1.0; N];

        let mut flat = 0;
        while flat < N {
            let mut r = flat;
            let mut d = 0;
            while d < D {
                let i = r % M;
                r /= M;
                pts[d][flat] = *unsafe { xs.get_unchecked(i, 0) };
                ws[flat] *= *unsafe { weights.get_unchecked(i, 0) };
                d += 1;
            }
            flat += 1;
        }

        Self {
            weights: Vec::from_array([ws]),
            xs: Mat::from_array(pts),
        }
    }
}

impl<const N: usize, T> Quadrature<1, T> for GaussQuadrature<N, T>
where
    T: Clone + Zero + Sub<Output = T> + One + Div<Output = T> + MulAddAssign,
{
    fn integrate(&self, mut f: impl FnMut(Mat<1, 1, T>) -> T) -> T {
        Zip::from(&self.weights)
            .and(&self.xs)
            .fold(T::zero(), |acc, w, x| {
                let mut f = f(v![x.clone()]);
                f.mul_add_assign(w.clone(), acc);
                f
            })
    }
}

impl<const D: usize, const N: usize, T> Quadrature<D, T> for TensorGaussQuadrature<D, N, T>
where
    T: Clone + Zero + Sub<Output = T> + One + Div<Output = T> + MulAddAssign,
{
    fn integrate(&self, mut f: impl FnMut(Mat<1, D, T>) -> T) -> T {
        Zip::from(&self.weights)
            .and(&self.xs)
            .fold(T::zero(), move |acc, w, xs| {
                let mut f = f(v![xs.clone()]);
                f.mul_add_assign(w.clone(), acc);
                f
            })
    }
}

impl<const D: usize, T, Q> LinearMap<T, Q, D> {
    #[inline]
    fn map(&self, x: &mut T, d: usize)
    where
        T: Clone + One + Add<Output = T> + Sub<Output = T> + Div<Output = T> + MulAddAssign,
    {
        let (a, b) = (self.from[d].clone(), self.to[d].clone());
        let two = T::one() + T::one();
        let half_len = (b.clone() - a.clone()) / two.clone();
        let mid_point = (a + b) / two;

        x.mul_add_assign(half_len, mid_point);
    }

    pub fn new(inner: Q, from: impl Into<Vec<D, T>>, to: impl Into<Vec<D, T>>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            inner,
        }
    }
}

impl<const D: usize, T, Q> Quadrature<D, T> for LinearMap<T, Q, D>
where
    Q: Quadrature<D, T>,
    T: Clone
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + MulAddAssign
        + Mul<Output = T>,
{
    fn integrate(&self, mut f: impl FnMut(Mat<1, D, T>) -> T) -> T {
        let two = T::one() + T::one();
        let factor = self.from.zip_fold(&self.to, T::one(), |acc, a, b| {
            let half_len = (b - a) / two.clone();
            acc * half_len
        });

        factor
            * self.inner.integrate(move |mut xs| {
                for (d, x) in xs.iter_mut().enumerate() {
                    self.map(x, d)
                }
                f(xs)
            })
    }
}

#[cfg(test)]
mod tests {
    use core::{f64::consts::PI, ops::Neg};

    use matrixcompare::assert_scalar_eq;

    use crate::element::quadrature::{
        GaussQuadrature, LinearMap, Quadrature, TensorGaussQuadrature, cos_approx_newton,
    };

    #[test]
    fn test_1d_integration() {
        static QUAD: GaussQuadrature<19> = GaussQuadrature::<19>::new(10);
        let area = LinearMap::new(&QUAD, [0.0], [10.0]).integrate(|xs| {
            let x = xs.x;
            2.0 / PI.sqrt() * (-x * x).exp()
        });
        assert_scalar_eq!(area, 1.0, comp = abs, tol = 1e-7);
    }

    #[test]
    fn test_2d_integration() {
        static QUAD: TensorGaussQuadrature<2, 361> = TensorGaussQuadrature::new::<19>(10);

        let area = LinearMap::new(&QUAD, [0.0, 0.0], [10.0, 10.0])
            .integrate(|xs| 2.0 / PI.sqrt() * (xs.x + xs.y).powi(2).neg().exp());

        assert_scalar_eq!(area, 0.56418958, comp = abs, tol = 1e-7);
    }

    #[test]
    fn test_cos_approx() {
        let x = PI / 3.0;
        let cos_60 = x.cos();
        let cos_approx_60 = cos_approx_newton(x, 20);

        assert_scalar_eq!(&cos_60, &cos_approx_60, comp = abs, tol = 1e-7);
    }
}
