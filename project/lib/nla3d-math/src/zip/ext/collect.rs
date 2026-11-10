use generic_array::typenum::Const;

use crate::dense::{DMat, Lower, Mat, MatSym, SymKind, Upper};
use crate::zip::{ElementwiseConstructible, KnownShape, MatSymLength, Producer, Zip};

pub trait CollectIntoExt: Producer
where
    Self::Shape: KnownShape,
{
    fn into_dyn(self) -> DMat<Self::Item>
    where
        DMat<Self::Item>: ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>;

    fn into_static<const N: usize, const M: usize>(self) -> Mat<N, M, Self::Item>
    where
        Mat<N, M, Self::Item>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>;

    #[inline]
    fn into_lower<const N: usize>(self) -> MatSym<N, Self::Item, Lower>
    where
        MatSym<N, Self::Item, Lower>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength,
    {
        self.into_symmetric()
    }

    #[inline]
    fn into_upper<const N: usize>(self) -> MatSym<N, Self::Item, Upper>
    where
        MatSym<N, Self::Item, Upper>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength,
    {
        self.into_symmetric()
    }

    fn into_symmetric<const N: usize, K: SymKind>(self) -> MatSym<N, Self::Item, K>
    where
        MatSym<N, Self::Item, K>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength;
}

impl<T: Producer<Shape: KnownShape>> CollectIntoExt for T {
    #[inline]
    fn into_dyn(self) -> DMat<Self::Item>
    where
        DMat<Self::Item>: ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
    {
        Zip::from(self).collect_into()
    }

    #[inline]
    fn into_static<const N: usize, const M: usize>(self) -> Mat<N, M, Self::Item>
    where
        Mat<N, M, Self::Item>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
    {
        Zip::from(self).collect_into()
    }

    #[inline]
    fn into_symmetric<const N: usize, K: SymKind>(self) -> MatSym<N, Self::Item, K>
    where
        MatSym<N, Self::Item, K>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength,
    {
        Zip::from(self).collect_into()
    }
}
