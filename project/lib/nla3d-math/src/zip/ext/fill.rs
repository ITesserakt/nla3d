use crate::zip::{DomainIndex, Free, Full, Producer, Static};

pub struct Fill<T>(pub T);

pub struct FillStatic<const N: usize, const M: usize, T>(pub T);

unsafe impl<T: Clone> Producer for Fill<T> {
    type Domain = Full;
    type Shape = Free;
    type Item = T;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Free
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, _: DomainIndex) -> Self::Item {
        cursor.0.clone()
    }
}

unsafe impl<const N: usize, const M: usize, T: Clone> Producer for FillStatic<N, M, T> {
    type Domain = Full;
    type Shape = Static<N, M>;
    type Item = T;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, _: DomainIndex) -> Self::Item {
        cursor.0.clone()
    }
}
