mod construct;
mod drive;
mod ext;
mod layout;
mod producer;
mod tuple;

use core::marker::PhantomData;

pub use construct::{ElementwiseConstructible, Nothing};
pub use ext::{CollectIntoExt, OwnedExt, TriExt};
pub use layout::{Any, Dynamic, Fits, Join, Layout, Static, Symmetric};
pub use producer::{Consume, Fill, FillStatic, Producer, RawMut, ToOwnedProducer, Tri};
pub use tuple::Push;

#[cfg(feature = "rand")]
pub use ext::RandomConstructible;

pub struct Zip<T, L>(T, PhantomData<L>);

impl<T> Zip<(T,), T::Layout>
where
    T: Producer,
{
    pub const fn from(value: T) -> Self {
        Self((value,), PhantomData)
    }
}

impl<T, L: Layout> Zip<T, L> {
    pub fn and<U>(self, value: U) -> Zip<T::Output, L::Out>
    where
        T: Push<U>,
        L: Join<U::Layout>,
        U: Producer,
    {
        Zip(self.0.push(value), PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use core::mem::ManuallyDrop;

    use matrixcompare::assert_matrix_eq;
    use rand::SeedableRng;

    use crate::dense::Mat;
    use crate::zip::RandomConstructible;
    use crate::zip::TriExt;
    use crate::zip::Zip;
    use crate::zip::producer::Consume;
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
        let mut a = m![[1, 2], [3, 4]].into_dyn();
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

    #[test]
    fn test_triangular_traverse() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut m = MatSym::<3>::random_uniform((3, 3), &mut rng);

        let mut visited = 0;
        Zip::from(&mut m).for_each(|x| {
            visited += 1;
            *x = match *x > 0.5 {
                true => 2.0,
                false => -1.0,
            };
        });

        let product = Zip::from(&m).fold(1.0, std::ops::Mul::mul);
        let product_lower = Zip::from(m.lower()).fold(1.0, std::ops::Mul::mul);
        assert_eq!(visited, 6);
        assert_eq!(product, 32.0);
        assert_eq!(product_lower, -8.0);
    }
}
