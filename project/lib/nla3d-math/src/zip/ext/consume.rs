use core::mem::ManuallyDrop;

use generic_array::GenericArray;
use generic_array::typenum::Const;

use crate::dense::{Mat, MatSym, MatSymLength, SymKind};
use crate::zip::{DomainIndex, Full, Producer, Static, Triangular};

pub trait ConsumeExt: Sized {
    type Producer: Producer;

    fn consume(self) -> Self::Producer;
}

impl<T> ConsumeExt for T
where
    ConsumeProducer<T>: Producer,
{
    type Producer = ConsumeProducer<Self>;

    fn consume(self) -> ConsumeProducer<Self> {
        ConsumeProducer(self)
    }
}

pub struct ConsumeProducer<T>(pub T);

pub struct ConsumeCursor<const N: usize, const M: usize, T> {
    mat: ManuallyDrop<Mat<N, M, T>>,
    taken: [[bool; N]; M],
}

pub struct SymConsumeCursor<const N: usize, T, K>
where
    Const<N>: MatSymLength,
{
    mat: ManuallyDrop<MatSym<N, T, K>>,
    taken: GenericArray<bool, <Const<N> as MatSymLength>::Length>,
}

unsafe impl<const N: usize, const M: usize, T> Producer for ConsumeProducer<Mat<N, M, T>> {
    type Domain = Full;
    type Shape = Static<N, M>;
    type Item = T;
    type Cursor = ConsumeCursor<N, M, T>;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        ConsumeCursor {
            mat: ManuallyDrop::new(self.0),
            taken: [[false; N]; M],
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let value = unsafe { std::ptr::read(cursor.mat.get_unchecked(*index.i, *index.j)) };
        cursor.taken[*index.j][*index.i] = true;
        value
    }
}

unsafe impl<const N: usize, T, K> Producer for ConsumeProducer<MatSym<N, T, K>>
where
    Const<N>: MatSymLength,
    K: SymKind,
{
    type Domain = Triangular<K>;
    type Shape = Static<N, N>;
    type Item = T;
    type Cursor = SymConsumeCursor<N, T, K>;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        Triangular::default()
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        SymConsumeCursor {
            mat: ManuallyDrop::new(self.0),
            taken: GenericArray::default(),
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let reference = unsafe { cursor.mat.get_unchecked_flat(index.pos) };
        let value = unsafe { std::ptr::read(reference) };
        cursor.taken[index.pos] = true;
        value
    }
}

impl<const N: usize, const M: usize, T> Drop for ConsumeCursor<N, M, T> {
    fn drop(&mut self) {
        if const { !std::mem::needs_drop::<T>() } {
            return;
        }

        let base = self.mat.as_mut_ptr().cast::<T>();
        self.taken
            .as_flattened()
            .iter()
            .enumerate()
            .filter(|&(_, taken)| !taken)
            .map(|(offset, _)| unsafe { base.add(offset) })
            .for_each(|it| unsafe { it.drop_in_place() });
    }
}
