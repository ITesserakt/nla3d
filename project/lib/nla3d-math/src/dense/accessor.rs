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
