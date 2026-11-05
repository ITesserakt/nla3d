use faer::mat::{AsMatMut, AsMatRef};
use faer::{MatMut, MatRef};
use std::mem::MaybeUninit;
use std::ops::*;

use crate::dense::zip::{Consume, Zip};
use crate::dense::{DMat, Mat, Vec};
use crate::num_traits::{ConstOne, ConstZero, One, Zero};

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
        let mut array = self.0.0.map(MaybeUninit::new);
        let mut out = DMat::from_fn(N, M, |_, _| const { MaybeUninit::uninit() });

        for i in 0..M {
            let col = &mut array[i];
            let col: &mut [_] = col.as_mut();

            for j in 0..N {
                let slot = &mut col[j];
                let value = unsafe { slot.assume_init_read() };
                out[(j, i)].write(value);
            }
        }

        unsafe { std::mem::transmute(out) }
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

macro_rules! impl_operator {
    ($trait_name:ident $trait_fn:ident) => {
        impl<const N: usize, const M: usize, T> $trait_name for Mat<N, M, T>
        where
            T: $trait_name,
        {
            type Output = Mat<N, M, T::Output>;

            #[inline]
            fn $trait_fn(self, rhs: Self) -> Self::Output {
                Zip::from(Consume(self))
                    .and(Consume(rhs))
                    .map(|x, y| x.$trait_fn(y))
            }
        }

        impl<'a, const N: usize, const M: usize, T> $trait_name<Mat<N, M, T>> for &'a Mat<N, M, T>
        where
            &'a T: $trait_name<T>,
        {
            type Output = Mat<N, M, <&'a T as $trait_name<T>>::Output>;

            #[inline]
            fn $trait_fn(self, rhs: Mat<N, M, T>) -> Self::Output {
                Zip::from(self).and(Consume(rhs)).map(|x, y| x.$trait_fn(y))
            }
        }

        impl<'a, const N: usize, const M: usize, T> $trait_name<&'a Mat<N, M, T>> for Mat<N, M, T>
        where
            T: $trait_name<&'a T>,
        {
            type Output = Mat<N, M, <T as $trait_name<&'a T>>::Output>;

            #[inline]
            fn $trait_fn(self, rhs: &'a Mat<N, M, T>) -> Self::Output {
                Zip::from(Consume(self)).and(rhs).map(|x, y| x.$trait_fn(y))
            }
        }

        impl<'a, const N: usize, const M: usize, T> $trait_name<&'a Mat<N, M, T>>
            for &'a Mat<N, M, T>
        where
            &'a T: $trait_name<&'a T>,
        {
            type Output = Mat<N, M, <&'a T as $trait_name<&'a T>>::Output>;

            #[inline]
            fn $trait_fn(self, rhs: &'a Mat<N, M, T>) -> Self::Output {
                Zip::from(self).and(rhs).map(|x, y| x.$trait_fn(y))
            }
        }

        impl<const N: usize, const M: usize, T> $trait_name<T> for Mat<N, M, T>
        where
            T: $trait_name + Clone,
        {
            type Output = Mat<N, M, T::Output>;

            #[inline]
            fn $trait_fn(self, rhs: T) -> Self::Output {
                let mut array = MaybeUninit::uninit();
                let col: &mut [_] = array.as_mut();

                for i in 0..M {
                    let mut array = MaybeUninit::uninit();
                    let row: &mut [_] = array.as_mut();

                    for j in 0..N {
                        row[j].write(self.0.0[i][j].clone().$trait_fn(rhs.clone()));
                    }

                    col[i].write(unsafe { array.assume_init() });
                }

                Mat(ArrayStorage(unsafe { array.assume_init() }))
            }
        }

        impl<'a, const N: usize, const M: usize, T> $trait_name<&'a T> for Mat<N, M, T>
        where
            T: $trait_name<&'a T> + Clone,
        {
            type Output = Mat<N, M, T::Output>;

            #[inline]
            fn $trait_fn(self, rhs: &'a T) -> Self::Output {
                let mut array = MaybeUninit::uninit();
                let col: &mut [_] = array.as_mut();

                for i in 0..M {
                    let mut array = MaybeUninit::uninit();
                    let row: &mut [_] = array.as_mut();

                    for j in 0..N {
                        row[j].write(self.0.0[i][j].clone().$trait_fn(rhs));
                    }

                    col[i].write(unsafe { array.assume_init() });
                }

                Mat(ArrayStorage(unsafe { array.assume_init() }))
            }
        }

        impl<'a, const N: usize, const M: usize, T> $trait_name<T> for &'a Mat<N, M, T>
        where
            &'a T: $trait_name<T>,
            T: Clone,
        {
            type Output = Mat<N, M, <&'a T as $trait_name<T>>::Output>;

            #[inline]
            fn $trait_fn(self, rhs: T) -> Self::Output {
                let mut array = MaybeUninit::uninit();
                let col: &mut [_] = array.as_mut();

                for i in 0..M {
                    let mut array = MaybeUninit::uninit();
                    let row: &mut [_] = array.as_mut();

                    for j in 0..N {
                        row[j].write(self.0.0[i][j].$trait_fn(rhs.clone()));
                    }

                    col[i].write(unsafe { array.assume_init() });
                }

                Mat(ArrayStorage(unsafe { array.assume_init() }))
            }
        }

        impl<'a, const N: usize, const M: usize, T> $trait_name<&'a T> for &'a Mat<N, M, T>
        where
            &'a T: $trait_name,
        {
            type Output = Mat<N, M, <&'a T as $trait_name>::Output>;

            #[inline]
            fn $trait_fn(self, rhs: &'a T) -> Self::Output {
                let mut array = MaybeUninit::uninit();
                let col: &mut [_] = array.as_mut();

                for i in 0..M {
                    let mut array = MaybeUninit::uninit();
                    let row: &mut [_] = array.as_mut();

                    for j in 0..N {
                        row[j].write(self.0.0[i][j].$trait_fn(rhs));
                    }

                    col[i].write(unsafe { array.assume_init() });
                }

                Mat(ArrayStorage(unsafe { array.assume_init() }))
            }
        }
    };
}

macro_rules! impl_assign_operator {
    ($trait_name:ident $trait_fn:ident) => {
        impl<const N: usize, const M: usize, T> $trait_name for Mat<N, M, T>
        where
            T: $trait_name,
        {
            #[inline]
            fn $trait_fn(&mut self, rhs: Self) {
                Zip::from(self)
                    .and(Consume(rhs))
                    .for_each(|y, x| y.$trait_fn(x))
            }
        }

        impl<const N: usize, const M: usize, T> $trait_name<&Mat<N, M, T>> for Mat<N, M, T>
        where
            T: for<'a> $trait_name<&'a T>,
        {
            #[inline]
            fn $trait_fn(&mut self, rhs: &Mat<N, M, T>) {
                Zip::from(self)
                    .and(rhs)
                    .for_each(|y, x| y.$trait_fn(x))
            }
        }

        impl<const N: usize, const M: usize, T> $trait_name<T> for Mat<N, M, T>
        where
            T: $trait_name + Clone,
        {
            #[inline]
            fn $trait_fn(&mut self, rhs: T) {
                for j in 0..N {
                    for i in 0..M {
                        self.0.0[i][j].$trait_fn(rhs.clone());
                    }
                }
            }
        }

        impl<const N: usize, const M: usize, T> $trait_name<&T> for Mat<N, M, T>
        where
            T: for<'a> $trait_name<&'a T>,
        {
            #[inline]
            fn $trait_fn(&mut self, rhs: &T) {
                for j in 0..N {
                    for i in 0..M {
                        self.0.0[i][j].$trait_fn(rhs);
                    }
                }
            }
        }
    };
}

impl_operator!(Add add);
impl_operator!(Sub sub);
impl_operator!(Mul mul);
impl_operator!(Div div);
impl_assign_operator!(AddAssign add_assign);
impl_assign_operator!(SubAssign sub_assign);
impl_assign_operator!(MulAssign mul_assign);
impl_assign_operator!(DivAssign div_assign);

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

#[cfg(test)]
mod tests {
    use matrixcompare::assert_matrix_eq;

    use crate::{FaerCompare, m, v};

    #[test]
    fn test_vec_addition() {
        let a = v![1, 2, 3];
        let b = v![4, 5, 6];
        let c = v![5, 7, 9];

        assert_matrix_eq!(c, a + b);
        assert_matrix_eq!(c, &a + b);
        assert_matrix_eq!(c, a + &b);
        assert_matrix_eq!(c, &a + &b);
    }

    #[test]
    fn test_vec_assign_addition() {
        let a = v![1, 2, 3];
        let mut b = v![0, 0, 0];

        b += a;
        assert_matrix_eq!(b, a);
    }

    #[test]
    fn test_into_dyn() {
        let a = m![[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let b = a.into_dyn();

        assert_matrix_eq!(a, FaerCompare(b));
    }
}
