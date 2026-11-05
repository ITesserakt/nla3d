mod construct;
mod drive;
mod producer;
mod reduce;
mod tuple;

pub use construct::{ElementwiseConstructible, Nothing};
pub use producer::{Consume, Fill, Producer, RawMut};
pub use reduce::{ReduceToCommonMatrix, ReducedToCommonMatrix, RemoveRef, Zippable};
pub use tuple::{Push, Tuple};

pub struct Zip<T>(T);

impl<T> Zip<(T,)> {
    pub const fn from(value: T) -> Self
    where
        T: Producer,
    {
        Self((value,))
    }
}

impl<T> Zip<T> {
    pub fn and<U>(self, value: U) -> Zip<T::Output>
    where
        T: Tuple<Tail: RemoveRef> + Push<U>,
        U: Producer + RemoveRef,
        <T::Tail as RemoveRef>::Output: Zippable<U::Output>,
    {
        Zip(self.0.push(value))
    }
}

#[cfg(test)]
mod tests {
    use core::mem::ManuallyDrop;

use matrixcompare::assert_matrix_eq;

    use crate::dense::Mat;
    use crate::dense::zip::Zip;
    use crate::dense::zip::producer::Consume;
    use crate::{dense::MatSym, m};

    const A: Mat<3, 3, usize> = m![[1, 2, 3], [2, 3, 4], [5, 6, 7]];

    fn b() -> MatSym<3, f64> {
        MatSym::from_mat(m![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    }

    #[test]
    fn test_zip_fold() {
        let sum = Zip::from(&A)
            .and(&b())
            .fold(0.0, |acc, x, y| acc + (*x as f64) * *y);

        assert_eq!(sum, 4.0);
    }

    #[test]
    fn test_zip_for_each() {
        let mut total_non_zero = 0;
        Zip::from(&A).and(&b()).for_each(|x, y| {
            if (*x as f64) * *y != 0.0 {
                total_non_zero += 1;
            }
        });

        assert_eq!(total_non_zero, 2);
    }

    #[test]
    fn test_zip_map_different_types() {
        let out = Zip::from(&A).and(&b()).map(|x, y| (*x as f64) * *y);

        assert_matrix_eq!(out, m![[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_zip_map_same_type() {
        let out = Zip::from(&b()).and(&b()).map(|x, y| (*x * *y) as u8);

        assert_matrix_eq!(out, m![[0, 1, 0], [1, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn for_each_mut_items_can_be_retained() {
        let mut a = m![[1, 2], [3, 4]];
        let mut v = Vec::new();
        Zip::from(&mut a).for_each(|x| v.push(x));
        for x in v {
            *x += 1;
        }
    }

    #[test]
    fn change_output_type() {
        let a = m![[1, 2], [3, 4]];
        let b = Zip::from(Consume(a)).map(|x| x);

        assert_matrix_eq!(a, b);
    }

    #[test]
    fn test_panicing() {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        struct IncOnDrop(u32);

        impl Drop for IncOnDrop {
            fn drop(&mut self) {
                COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        
        let a = m![[IncOnDrop(1), IncOnDrop(2)], [IncOnDrop(3), IncOnDrop(4)]];

        let result = std::panic::catch_unwind(|| {
            Zip::from(Consume(a)).for_each(|x| {
                let x = ManuallyDrop::new(x);
                if x.0 == 3 {
                    panic!("Test");
                }
            });
        });

        assert_eq!(COUNTER.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(matches!(result, Err(_)));
    }
}
