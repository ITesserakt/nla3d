use crate::zip::{Domain, DomainIndex, Free, Producer};

pub struct FromFn<D, F>(D, F);

impl<D, F> FromFn<D, F> {
    pub fn new<T>(domain: D, f: F) -> Self
    where
        F: FnMut(DomainIndex) -> T,
    {
        Self(domain, f)
    }
}

unsafe impl<D, T, F> Producer for FromFn<D, F>
where
    D: Domain,
    F: FnMut(DomainIndex) -> T,
{
    type Domain = D;
    type Shape = Free;
    type Item = T;
    type Cursor = F;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        self.0.clone()
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Free
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self.1
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        cursor(index)
    }
}
