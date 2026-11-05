use core::mem::MaybeUninit;
use std::fmt::Debug;
use std::fmt::Write;
use std::marker::PhantomData;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;

use generic_array::sequence::GenericSequence;
use generic_array::typenum::Const;
use generic_array::{ArrayLength, GenericArray};

use crate::dense::Mat;
use crate::dense::MatSym;
use crate::num_traits::Zero;

pub struct Upper;
pub struct Lower;

pub struct SymArrayStorage<T, N: MatSymLength, Kind> {
    data: GenericArray<T, N::Length>,
    _phantom: PhantomData<Kind>,
}

pub trait MatSymLength {
    type Length: ArrayLength;
}

mod mat_sym_length_impls {
    macro_rules! impl_mat_sym_length_for {
        ($($n:literal$(,)?)*) => {
            $(
                impl_mat_sym_length_for!(inner $n);
                impl_mat_sym_length_for!(inner $n + 1);
                impl_mat_sym_length_for!(inner $n + 2);
                impl_mat_sym_length_for!(inner $n + 3);
            )*
        };
        (inner $($t:tt)+) => {
            impl super::MatSymLength for generic_array::typenum::Const<{$($t)+}> {
                type Length = <
                    generic_array::typenum::Const<{($($t)+) * (($($t)+) + 1) / 2}> as generic_array::typenum::ToUInt
                >::Output;
            }
        };
    }

    impl_mat_sym_length_for!(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40);
}

impl<T, N: MatSymLength, Kind> Deref for SymArrayStorage<T, N, Kind> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, N: MatSymLength, Kind> DerefMut for SymArrayStorage<T, N, Kind> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

pub trait SymKind {
    fn flat_index(side: usize, i: usize, j: usize) -> usize;

    fn wide_index(side: usize, k: usize) -> (usize, usize);
}

impl SymKind for Upper {
    #[inline]
    fn flat_index(side: usize, mut i: usize, mut j: usize) -> usize {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        i * (2 * side - i - 1) / 2 + j
    }

    #[inline]
    fn wide_index(side: usize, k: usize) -> (usize, usize) {
        let d = 2 * side + 1;
        let sqrt = (d * d - 8 * k).isqrt();
        let i = if sqrt * sqrt == d * d - 8 * k {
            (d - sqrt) / 2
        } else {
            (d - sqrt - 1) / 2
        };
        let j = i + k - i * (d - i) / 2;
        (i, j)
    }
}

impl SymKind for Lower {
    #[inline]
    fn flat_index(_: usize, mut i: usize, mut j: usize) -> usize {
        if i < j {
            std::mem::swap(&mut i, &mut j);
        }
        i * (i + 1) / 2 + j
    }

    #[inline]
    fn wide_index(_: usize, k: usize) -> (usize, usize) {
        let i = ((1 + 8 * k).isqrt() - 1) / 2;
        let j = k - i * (i + 1) / 2;
        (i, j)
    }
}

impl<const N: usize, T, Kind> MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (N, N)
    }

    #[inline]
    pub fn to_mat(&self) -> Mat<N, N, T>
    where
        T: Clone,
        Kind: SymKind,
    {
        let mut matrix = Mat::uninit();

        for (k, item) in self.0.iter().enumerate() {
            let (i, j) = Kind::wide_index(N, k);
            matrix[(i, j)].write(item.clone());
            matrix[(j, i)].write(item.clone());
        }

        unsafe { matrix.assume_init() }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T
    where
        Kind: SymKind,
    {
        let k = Kind::flat_index(N, i, j);
        unsafe { self.0.data.get_unchecked(k) }
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T
    where
        Kind: SymKind,
    {
        let k = Kind::flat_index(N, i, j);
        unsafe { self.0.data.get_unchecked_mut(k) }
    }

    #[inline]
    pub const fn uninit() -> MatSym<N, MaybeUninit<T>, Kind> {
        MatSym(SymArrayStorage {
            data: GenericArray::uninit(),
            _phantom: PhantomData,
        })
    }

    #[inline]
    pub fn from_mat(matrix: Mat<N, N, T>) -> Self
    where
        Kind: SymKind,
        T: Clone,
    {
        let mut out = Self::uninit();
        for (k, slot) in out.0.iter_mut().enumerate() {
            let (i, j) = Kind::wide_index(N, k);
            slot.write(matrix[(i, j)].clone());
        }
        unsafe { out.assume_init() }
    }
}

impl<const N: usize, T, Kind> MatSym<N, MaybeUninit<T>, Kind>
where
    Const<N>: MatSymLength,
{
    pub unsafe fn assume_init(self) -> MatSym<N, T, Kind> {
        MatSym(SymArrayStorage {
            data: unsafe { GenericArray::assume_init(self.0.data) },
            _phantom: PhantomData,
        })
    }
}

impl<const N: usize, T, Kind: SymKind> Index<(usize, usize)> for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[Kind::flat_index(N, index.0, index.1)]
    }
}

impl<const N: usize, T, Kind: SymKind> IndexMut<(usize, usize)> for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[Kind::flat_index(N, index.0, index.1)]
    }
}

impl<const N: usize, T: Zero, Kind> Zero for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    T: Clone,
{
    fn zero() -> Self {
        Self(SymArrayStorage {
            data: GenericArray::generate(|_| T::zero()),
            _phantom: PhantomData,
        })
    }

    fn is_zero(&self) -> bool {
        self.0.data.iter().all(|it| it.is_zero())
    }
}

impl<const N: usize, T, Kind> Add for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    T: Add<Output = T> + Clone,
{
    type Output = MatSym<N, T, Kind>;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.data.into_iter())
            .for_each(|(y, x)| {
                *y = y.clone().add(x);
            });
        self
    }
}

impl<const N: usize, T, Kind> PartialEq for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.data.eq(&other.0.data)
    }
}

impl<const N: usize, T, Kind> PartialEq<Mat<N, N, T>> for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    T: PartialEq,
    Kind: SymKind,
{
    fn eq(&self, other: &Mat<N, N, T>) -> bool {
        self.0.iter().enumerate().all(|(k, it)| {
            let (i, j) = Kind::wide_index(N, k);
            other[(i, j)].eq(it)
        })
    }
}

impl<const N: usize, T, Kind> PartialEq<MatSym<N, T, Kind>> for Mat<N, N, T>
where
    Const<N>: MatSymLength,
    T: PartialEq,
    Kind: SymKind,
{
    #[inline]
    fn eq(&self, other: &MatSym<N, T, Kind>) -> bool {
        other.eq(self)
    }
}

impl<const N: usize, T, Kind> Debug for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();

        for i in 0..N {
            list.entry_with(|f| {
                if i != 0 {
                    f.write_char('\n')?;
                    f.write_char(' ')?;
                }
                let mut list = f.debug_list();

                for j in 0..N {
                    list.entry(&self[(i, j)]);
                }

                list.finish()
            });
        }

        list.finish()
    }
}

impl<const N: usize, T, Kind> AddAssign for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.data.into_iter())
            .for_each(|(y, x)| {
                y.add_assign(x);
            });
    }
}

mod fmt_impls {
    use super::{MatSym, MatSymLength, SymKind};
    use generic_array::typenum::Const;
    use std::fmt;

    macro_rules! impl_fmt {
        ($trait: path, $fmt_str_without_precision: expr_2021, $fmt_str_with_precision: expr_2021) => {
            impl<T, const N: usize, Kind> $trait for MatSym<N, T, Kind>
            where
                T: Clone + $trait,
                Const<N>: MatSymLength,
                Kind: SymKind,
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

#[cfg(feature = "compare")]
const _: () = {
    use matrixcompare::{DenseAccess, Matrix};

    impl<const N: usize, T, Kind> Matrix<T> for MatSym<N, T, Kind>
    where
        Const<N>: MatSymLength,
        Kind: SymKind,
        T: Clone,
    {
        #[inline]
        fn rows(&self) -> usize {
            N
        }

        #[inline]
        fn cols(&self) -> usize {
            N
        }

        fn access(&self) -> matrixcompare::Access<'_, T> {
            matrixcompare::Access::Dense(self)
        }
    }

    impl<const N: usize, T, Kind> DenseAccess<T> for MatSym<N, T, Kind>
    where
        Const<N>: MatSymLength,
        Kind: SymKind,
        T: Clone,
    {
        fn fetch_single(&self, row: usize, col: usize) -> T {
            self[(row, col)].clone()
        }
    }
};
