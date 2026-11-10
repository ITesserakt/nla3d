use core::ptr::NonNull;
use std::mem::MaybeUninit;
use std::ops::*;

use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::ext::ComplexFieldExt;
use faer::traits::{ComplexField, RealField};
use faer::{MatMut, MatRef};
use num_traits::{ConstOne, ConstZero, One, Zero};

use crate::dense::{DMat, Mat, SingularMatrix, Vec};
use crate::m;
use crate::zip::ext::{CollectIntoExt, ConsumeExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ArrayStorage<T, const R: usize, const C: usize>(pub [[T; R]; C]);

impl<const N: usize, const M: usize, T> Mat<N, M, T> {
    #[inline]
    pub fn view(&self) -> MatRef<'_, T> {
        MatRef::from_column_major_array(&self.0.0)
    }

    #[inline]
    pub fn view_mut(&mut self) -> MatMut<'_, T> {
        MatMut::from_column_major_array_mut(&mut self.0.0)
    }

    #[inline]
    pub const fn from_array(array: [[T; N]; M]) -> Self {
        Self(ArrayStorage(array))
    }

    #[inline]
    pub fn from_elem(elem: T) -> Self
    where
        T: Clone,
    {
        Self(ArrayStorage(std::array::repeat(std::array::repeat(elem))))
    }

    #[inline]
    pub fn into_dyn(self) -> DMat<T> {
        self.consume().into_dyn()
    }

    #[inline]
    pub const fn uninit() -> Mat<N, M, MaybeUninit<T>> {
        Mat(ArrayStorage(
            [const { [const { MaybeUninit::uninit() }; N] }; M],
        ))
    }

    #[inline(always)]
    pub const unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        &self.0.0[j][i]
    }

    #[inline(always)]
    pub const unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.0.0[j][i]
    }

    #[inline]
    pub fn row_iter(&self) -> impl ExactSizeIterator<Item = &[T; N]> {
        self.0.0.iter()
    }

    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (N, M)
    }

    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> NonNull<[T]> {
        NonNull::from_mut(self.0.0.as_flattened_mut())
    }
}

impl<const N: usize, T: ComplexField> Mat<N, N, T> {
    pub fn determinant(&self) -> T {
        fn cast<const N: usize, const W: usize, const H: usize, T>(
            matrix: &Mat<W, H, T>,
        ) -> &Mat<N, N, T> {
            assert!(N == W && N == H);
            unsafe { std::mem::transmute(matrix) }
        }

        match N {
            0 => T::one_impl(),
            1 => {
                let m = cast::<1, _, _, _>(self).deref().clone();
                m.x
            }
            2 => {
                let m = cast::<2, _, _, _>(self).deref().clone();
                m.xx * m.yy - m.xy * m.yx
            }
            3 => {
                let m = cast::<3, _, _, _>(self).deref().clone();

                m.xx * (m.yy.clone() * m.zz.clone() - m.yz.clone() * m.zy.clone())
                    - m.xy * (m.yx.clone() * m.zz - m.yz * m.zx.clone())
                    + m.xz * (m.yx * m.zy - m.zx * m.yy)
            }
            _ => self.view().determinant(),
        }
    }
}

impl<T: RealField> Mat<0, 0, T> {
    pub const fn inv(&self) -> Self {
        Self::from_array([])
    }
}

impl<T: RealField> Mat<1, 1, T> {
    #[inline]
    pub fn inv(&self) -> Self {
        self.try_inv().unwrap()
    }

    #[inline]
    pub fn try_inv(&self) -> Result<Self, SingularMatrix> {
        let [[x]] = &self.0.0;
        if x.is_zero() {
            return Err(SingularMatrix);
        }
        Ok(m![[x.recip()]])
    }
}

impl<T: RealField> Mat<2, 2, T> {
    #[inline]
    pub fn inv(&self) -> Self {
        self.try_inv().unwrap()
    }

    #[inline]
    pub fn try_inv(&self) -> Result<Self, SingularMatrix> {
        let det = self.determinant();
        if det.is_zero() {
            return Err(SingularMatrix);
        }
        let inv_det = det.recip();
        let adjugate = m![
            [self.yy.clone(), -self.xy.clone()],
            [-self.yx.clone(), self.xx.clone()]
        ];

        Ok(adjugate * inv_det)
    }
}

impl<T: RealField> Mat<3, 3, T> {
    #[inline]
    pub fn inv(&self) -> Self {
        self.try_inv().unwrap()
    }

    #[inline]
    pub fn try_inv(&self) -> Result<Self, SingularMatrix> {
        let det = self.determinant();
        if det.is_zero() {
            return Err(SingularMatrix);
        }
        let inv_det = det.recip();

        let adjugate = m![
            [
                self[(1, 1)].clone() * self[(2, 2)].clone()
                    - self[(1, 2)].clone() * self[(2, 1)].clone(),
                self[(0, 2)].clone() * self[(2, 1)].clone()
                    - self[(0, 1)].clone() * self[(2, 2)].clone(),
                self[(0, 1)].clone() * self[(1, 2)].clone()
                    - self[(0, 2)].clone() * self[(1, 1)].clone()
            ],
            [
                self[(1, 2)].clone() * self[(2, 0)].clone()
                    - self[(1, 0)].clone() * self[(2, 2)].clone(),
                self[(0, 0)].clone() * self[(2, 2)].clone()
                    - self[(0, 2)].clone() * self[(2, 0)].clone(),
                self[(0, 2)].clone() * self[(1, 0)].clone()
                    - self[(0, 0)].clone() * self[(1, 2)].clone()
            ],
            [
                self[(1, 0)].clone() * self[(2, 1)].clone()
                    - self[(1, 1)].clone() * self[(2, 0)].clone(),
                self[(0, 1)].clone() * self[(2, 0)].clone()
                    - self[(0, 0)].clone() * self[(2, 1)].clone(),
                self[(0, 0)].clone() * self[(1, 1)].clone()
                    - self[(0, 1)].clone() * self[(1, 0)].clone()
            ]
        ];

        Ok(adjugate * inv_det)
    }
}

impl<const N: usize, const M: usize, T> Mat<N, M, MaybeUninit<T>> {
    #[inline]
    pub unsafe fn assume_init(self) -> Mat<N, M, T> {
        Mat(ArrayStorage(
            self.0.0.map(|it| it.map(|it| unsafe { it.assume_init() })),
        ))
    }
}

impl<const N: usize, T> IndexMut<usize> for Vec<N, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.0[0][index]
    }
}

impl<const N: usize, T> Index<usize> for Vec<N, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0.0[0][index]
    }
}

impl<const N: usize, const M: usize, T> IndexMut<(usize, usize)> for Mat<N, M, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0.0[index.1][index.0]
    }
}

impl<const N: usize, const M: usize, T> Index<(usize, usize)> for Mat<N, M, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0.0[index.1][index.0]
    }
}

impl<const N: usize, const M: usize, T> Index<(usize, usize)> for &Mat<N, M, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0.0[index.1][index.0]
    }
}

impl<const N: usize, const M: usize, T> AsMatRef for Mat<N, M, T> {
    type T = T;

    type Rows = usize;

    type Cols = usize;

    type Owned = DMat<T>;

    #[inline]
    fn as_mat_ref(&self) -> MatRef<'_, Self::T, Self::Rows, Self::Cols> {
        MatRef::from_column_major_array(&self.0.0)
    }
}

impl<const N: usize, const M: usize, T> AsMatMut for Mat<N, M, T> {
    #[inline]
    fn as_mat_mut(&mut self) -> MatMut<'_, Self::T, Self::Rows, Self::Cols> {
        MatMut::from_column_major_array_mut(&mut self.0.0)
    }
}

mod ops_impls {
    use std::convert::identity;
    use std::ops::*;

    use crate::dense::Mat;
    use crate::zip::Zip;
    use crate::zip::ext::{ConsumeExt, Fill};
    use crate::{impl_assign_operator, impl_operator};

    impl_operator!(<;const N: usize, const M: usize; T> impl Ops for Mat<N, M, T> {
        scalar = T;
        output = Mat<N, M, O>;
        output scalar = O;
        handle self = ConsumeExt::consume;
        handle rhs = ConsumeExt::consume;
    });

    impl_operator!(<'a; const N: usize, const M: usize; T> impl Ops for &'a Mat<N, M, T> {
        scalar = &'a T;
        output = Mat<N, M, O>;
        output scalar = O;
        handle self = identity;
        handle rhs = identity;
    });

    impl_operator!(<'a; const N: usize, const M: usize; T> impl Ops<&'a Mat<N, M, T>> for Mat<N, M, T> {
        scalar = T, &'a T;
        output = Mat<N, M, O>;
        output scalar = O;
        handle self = ConsumeExt::consume;
        handle rhs = identity;
    });

    impl_operator!(<'a; const N: usize, const M: usize; T> impl Ops<Mat<N, M, T>> for &'a Mat<N, M, T> {
        scalar = &'a T, T;
        output = Mat<N, M, O>;
        output scalar = O;
        handle self = identity;
        handle rhs = ConsumeExt::consume;
    });

    impl_operator!(<;const N: usize, const M: usize; T> impl Ops<T> for Mat<N, M, T>
        where T: Clone
    {
        scalar = T;
        output = Mat<N, M, O>;
        output scalar = O;
        handle self = ConsumeExt::consume;
        handle rhs = Fill;
    });

    impl_operator!(<'a; const N: usize, const M: usize; T> impl Ops<T> for &'a Mat<N, M, T>
        where T: Clone
    {
        scalar = &'a T, T;
        output = Mat<N, M, O>;
        output scalar = O;
        handle self = identity;
        handle rhs = Fill;
    });

    impl_assign_operator!(<; const N: usize, const M: usize; T> impl Ops for Mat<N, M, T> {
        scalar = T;
        handle rhs = ConsumeExt::consume;
    });

    impl_assign_operator!(<'a; const N: usize, const M: usize; T> impl Ops<&'a Mat<N, M, T>> for Mat<N, M, T> {
        scalar = T, &'a T;
        handle rhs = identity;
    });

    impl_assign_operator!(<; const N: usize, const M: usize; T> impl Ops<T> for Mat<N, M, T>
        where T: Clone
    {
        scalar = T;
        handle rhs = Fill;
    });
}

impl<const N: usize, const M: usize, T> Zero for Mat<N, M, T>
where
    T: Zero + Clone,
{
    #[inline]
    fn zero() -> Self {
        Self(ArrayStorage(std::array::repeat(std::array::from_fn(
            |_| T::zero(),
        ))))
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.0.as_flattened().iter().all(|it| it.is_zero())
    }
}

impl<const N: usize, const M: usize, T: ConstZero + Clone> ConstZero for Mat<N, M, T> {
    const ZERO: Self = Mat(ArrayStorage([const { [T::ZERO; N] }; M]));
}

impl<const N: usize, const M: usize, T: One + Clone> One for Mat<N, M, T> {
    #[inline]
    fn one() -> Self {
        Self(ArrayStorage(std::array::repeat(std::array::from_fn(
            |_| T::one(),
        ))))
    }
}

impl<const N: usize, const M: usize, T: ConstOne + Clone> ConstOne for Mat<N, M, T> {
    const ONE: Self = Mat(ArrayStorage([const { [T::ONE; N] }; M]));
}

#[cfg(feature = "compare")]
impl<const N: usize, const M: usize, T> matrixcompare::Matrix<T> for Mat<N, M, T>
where
    T: Clone,
{
    #[inline]
    fn rows(&self) -> usize {
        N
    }

    #[inline]
    fn cols(&self) -> usize {
        M
    }

    #[inline]
    fn access(&self) -> matrixcompare::Access<'_, T> {
        matrixcompare::Access::Dense(self)
    }
}

#[cfg(feature = "compare")]
impl<const N: usize, const M: usize, T> matrixcompare::DenseAccess<T> for Mat<N, M, T>
where
    T: Clone,
{
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> T {
        self.0.0[col][row].clone()
    }
}

mod fmt_impls {
    use std::fmt;

    use super::Mat;

    macro_rules! impl_fmt {
        ($trait: path, $fmt_str_without_precision: expr_2021, $fmt_str_with_precision: expr_2021) => {
            impl<T, const N: usize, const M: usize> $trait for Mat<N, M, T>
            where
                T: Clone + $trait,
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    #[cfg(feature = "std")]
                    fn val_width<T: Clone + $trait>(val: &T, f: &mut fmt::Formatter<'_>) -> usize {
                        match f.precision() {
                            Some(precision) => format!($fmt_str_with_precision, val, precision)
                                .chars()
                                .count(),
                            None => format!($fmt_str_without_precision, val).chars().count(),
                        }
                    }

                    #[cfg(not(feature = "std"))]
                    fn val_width<T: Clone + $trait>(_: &T, _: &mut fmt::Formatter<'_>) -> usize {
                        4
                    }

                    let (nrows, ncols) = self.shape();

                    if nrows == 0 || ncols == 0 {
                        return write!(f, "[ ]");
                    }

                    let mut max_length = 0;

                    for i in 0..nrows {
                        for j in 0..ncols {
                            max_length = std::cmp::max(max_length, val_width(&self[(i, j)], f));
                        }
                    }

                    let max_length_with_space = max_length + 1;

                    writeln!(f)?;
                    writeln!(
                        f,
                        "  ┌ {:>width$} ┐",
                        "",
                        width = max_length_with_space * ncols - 1
                    )?;

                    for i in 0..nrows {
                        write!(f, "  │")?;
                        for j in 0..ncols {
                            let number_length = val_width(&self[(i, j)], f) + 1;
                            let pad = max_length_with_space - number_length;
                            write!(f, " {:>thepad$}", "", thepad = pad)?;
                            match f.precision() {
                                Some(precision) => {
                                    write!(f, $fmt_str_with_precision, (*self)[(i, j)], precision)?
                                }
                                None => write!(f, $fmt_str_without_precision, (*self)[(i, j)])?,
                            }
                        }
                        writeln!(f, " │")?;
                    }

                    writeln!(
                        f,
                        "  └ {:>width$} ┘",
                        "",
                        width = max_length_with_space * ncols - 1
                    )?;
                    writeln!(f)
                }
            }
        };
    }
    impl_fmt!(fmt::Display, "{}", "{:.1$}");
    impl_fmt!(fmt::LowerExp, "{:e}", "{:.1$e}");
    impl_fmt!(fmt::UpperExp, "{:E}", "{:.1$E}");
    impl_fmt!(fmt::Octal, "{:o}", "{:1$o}");
    impl_fmt!(fmt::LowerHex, "{:x}", "{:1$x}");
    impl_fmt!(fmt::UpperHex, "{:X}", "{:1$X}");
    impl_fmt!(fmt::Binary, "{:b}", "{:.1$b}");
    impl_fmt!(fmt::Pointer, "{:p}", "{:.1$p}");
}
