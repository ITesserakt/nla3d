use crate::dense::{Lower, Mat, MatSym, Upper};

macro_rules! define_or_use_struct {
    (<$($generic:ident$(, )?)*> new $($n:ident$(,)?)+ [$name:ident]) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name<$($generic, )*> {
            $(pub $n:T, )+
        }
    };
    (<$($generic:ident$(,)?)*> use $($n:ident$(,)?)+ [$name:ident]) => {

    }
}

macro_rules! define_accessor {
    (<$($generic:ident$(, )?)*> $matrix:ty => $($n:ident$(,)?)+ [$t:tt $name:ident]) => {
        define_or_use_struct!(<$($generic, )*> $t $($n, )+ [$name]);

        impl<$($generic, )*> std::ops::Deref for $matrix {
            type Target = $name<$($generic, )*>;

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { std::mem::transmute(&self.0) }
            }
        }

        impl<$($generic, )*> std::ops::DerefMut for $matrix {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { std::mem::transmute(&mut self.0) }
            }
        }
    };
}

define_accessor!(<T> Mat<1, 1, T> => x [new Access1]);

define_accessor!(<T> Mat<1, 2, T> => x, y [new Access2]);
define_accessor!(<T> Mat<2, 1, T> => x, y [use Access2]);
define_accessor!(<T> Mat<2, 2, T> => xx, yx, xy, yy [new Access22]);

define_accessor!(<T> Mat<1, 3, T> => x, y, z [new Access3]);
define_accessor!(<T> Mat<3, 1, T> => x, y, z [use Access3]);
define_accessor!(<T> Mat<3, 2, T> => xx, yx, zx, xy, yy, zy [new Access32]);
define_accessor!(<T> Mat<2, 3, T> => xx, yx, xy, yy, xz, yz [new Access23]);
define_accessor!(<T> Mat<3, 3, T> => xx, yx, zx, xy, yy, zy, xz, yz, zz [new Access33]);

define_accessor!(<T> MatSym<1, T, Lower> => x [use Access1]);
define_accessor!(<T> MatSym<1, T, Upper> => x [use Access1]);

define_accessor!(<T> MatSym<2, T, Upper> => xx, xy, yy [new AccessSym2]);
define_accessor!(<T> MatSym<2, T, Lower> => xx, xy, yy [use AccessSym2]);

define_accessor!(<T> MatSym<3, T, Upper> => xx, xy, xz, yy, yz, zz [new AccessSym3U]);
define_accessor!(<T> MatSym<3, T, Lower> => xx, xy, yy, xz, yz, zz [new AccessSym3L]);

#[cfg(test)]
mod tests {
    use crate::dense::{Lower, Mat, MatSym, Upper};
    use crate::zip::RandomConstructible;
    use rand::{SeedableRng, rngs::SmallRng};

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn test_accessors_are_valid_m33() {
        let m = Mat::<3, 3>::random_uniform((3, 3), &mut rng());

        assert_eq!(m[(0, 0)], m.xx);
        assert_eq!(m[(0, 1)], m.xy);
        assert_eq!(m[(0, 2)], m.xz);
        assert_eq!(m[(1, 0)], m.yx);
        assert_eq!(m[(1, 1)], m.yy);
        assert_eq!(m[(1, 2)], m.yz);
        assert_eq!(m[(2, 0)], m.zx);
        assert_eq!(m[(2, 1)], m.zy);
        assert_eq!(m[(2, 2)], m.zz);
    }

    #[test]
    fn test_accessors_are_valid_m32() {
        let m = Mat::<3, 2>::random_uniform((3, 2), &mut rng());

        assert_eq!(m[(0, 0)], m.xx);
        assert_eq!(m[(0, 1)], m.xy);
        assert_eq!(m[(1, 0)], m.yx);
        assert_eq!(m[(1, 1)], m.yy);
        assert_eq!(m[(2, 0)], m.zx);
        assert_eq!(m[(2, 1)], m.zy);
    }

    #[test]
    fn test_accessors_are_valid_m23() {
        let m = Mat::<2, 3>::random_uniform((2, 3), &mut rng());

        assert_eq!(m[(0, 0)], m.xx);
        assert_eq!(m[(0, 1)], m.xy);
        assert_eq!(m[(0, 2)], m.xz);
        assert_eq!(m[(1, 0)], m.yx);
        assert_eq!(m[(1, 1)], m.yy);
        assert_eq!(m[(1, 2)], m.yz);
    }

    #[test]
    fn test_accessors_are_valid_m22() {
        let m = Mat::<2, 2>::random_uniform((2, 2), &mut rng());

        assert_eq!(m[(0, 0)], m.xx);
        assert_eq!(m[(0, 1)], m.xy);
        assert_eq!(m[(1, 0)], m.yx);
        assert_eq!(m[(1, 1)], m.yy);
    }

    #[test]
    fn test_accessors_are_valid_ms1() {
        let mu = MatSym::<1, f64, Upper>::random_uniform((1, 1), &mut rng());
        let ml = MatSym::<1, f64, Lower>::random_uniform((1, 1), &mut rng());

        assert_eq!(mu[(0, 0)], mu.x);

        assert_eq!(ml[(0, 0)], ml.x);
    }

    #[test]
    fn test_accessors_are_valid_ms2() {
        let mu = MatSym::<2, f64, Upper>::random_uniform((2, 2), &mut rng());
        let ml = MatSym::<2, f64, Lower>::random_uniform((2, 2), &mut rng());

        assert_eq!(mu[(0, 0)], mu.xx);
        assert_eq!(mu[(0, 1)], mu.xy);
        assert_eq!(mu[(1, 0)], mu.xy);
        assert_eq!(mu[(1, 1)], mu.yy);

        assert_eq!(ml[(0, 0)], ml.xx);
        assert_eq!(ml[(0, 1)], ml.xy);
        assert_eq!(ml[(1, 0)], ml.xy);
        assert_eq!(ml[(1, 1)], ml.yy);
    }

    #[test]
    fn test_accessors_are_valid_ms3() {
        let mu = MatSym::<3, f64, Upper>::random_uniform((3, 3), &mut rng());
        let ml = MatSym::<3, f64, Lower>::random_uniform((3, 3), &mut rng());

        assert_eq!(mu[(0, 0)], mu.xx);
        assert_eq!(mu[(0, 1)], mu.xy);
        assert_eq!(mu[(0, 2)], mu.xz);
        assert_eq!(mu[(1, 0)], mu.xy);
        assert_eq!(mu[(1, 1)], mu.yy);
        assert_eq!(mu[(1, 2)], mu.yz);
        assert_eq!(mu[(2, 0)], mu.xz);
        assert_eq!(mu[(2, 1)], mu.yz);
        assert_eq!(mu[(2, 2)], mu.zz);

        println!("{ml}");
        println!("{:?}", ml.0);
        assert_eq!(ml[(0, 0)], ml.xx);
        assert_eq!(ml[(0, 1)], ml.xy);
        assert_eq!(ml[(0, 2)], ml.xz);
        assert_eq!(ml[(1, 0)], ml.xy);
        assert_eq!(ml[(1, 1)], ml.yy);
        assert_eq!(ml[(1, 2)], ml.yz);
        assert_eq!(ml[(2, 0)], ml.xz);
        assert_eq!(ml[(2, 1)], ml.yz);
        assert_eq!(ml[(2, 2)], ml.zz);
    }
}
