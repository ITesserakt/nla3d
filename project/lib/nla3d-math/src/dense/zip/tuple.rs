pub trait Tuple {
    type Body: Tuple;
    type Tail;

    fn split(self) -> (Self::Body, Self::Tail);
}

pub trait Push<T> {
    type Output: Tuple<Body = Self, Tail = T>;

    fn push(self, value: T) -> Self::Output;
}

macro_rules! impl_tuple {
    (closing $($n: ident: $t:ident$(,)?)*; $last_n:ident: $last:ident) => {
        impl<$($t, )* $last> Tuple for ($($t, )* $last) {
            type Body = ($($t, )*);
            type Tail = $last;

            #[inline(always)]
            fn split(self) -> (Self::Body, Self::Tail) {
                let ($($n, )* $last_n) = self;
                (($($n, )*), $last_n)
            }
        }
    };
    ($($n: ident: $t:ident$(,)?)*; $last_n:ident: $last:ident) => {
        impl<$($t, )* $last> Tuple for ($($t, )* $last) {
            type Body = ($($t, )*);
            type Tail = $last;

            #[inline(always)]
            fn split(self) -> (Self::Body, Self::Tail) {
                let ($($n, )* $last_n) = self;
                (($($n, )*), $last_n)
            }
        }

        impl<$($t, )* $last, T> Push<T> for ($($t, )* $last) {
            type Output = ($($t, )* $last, T);

            #[inline(always)]
            fn push(self, value: T) -> Self::Output {
                let ($($n, )* $last_n) = self;
                ($($n, )* $last_n, value)
            }
        }
    };
}

impl Tuple for () {
    type Body = ();
    type Tail = ();

    #[inline(always)]
    fn split(self) -> (Self::Body, Self::Tail) {
        ((), ())
    }
}

impl<T> Push<T> for () {
    type Output = (T,);

    #[inline(always)]
    fn push(self, value: T) -> Self::Output {
        (value,)
    }
}

impl<T1> Tuple for (T1,) {
    type Body = ();
    type Tail = T1;

    #[inline(always)]
    fn split(self) -> (Self::Body, Self::Tail) {
        ((), self.0)
    }
}

impl<T1, T> Push<T> for (T1,) {
    type Output = (T1, T);

    #[inline(always)]
    fn push(self, value: T) -> Self::Output {
        (self.0, value)
    }
}

impl_tuple!(a: T1; b: T2);
impl_tuple!(a: T1, b: T2; c: T3);
impl_tuple!(a: T1, b: T2, c: T3; d: T4);
impl_tuple!(a: T1, b: T2, c: T3, d: T4; e: T5);
impl_tuple!(closing a: T1, b: T2, c: T3, d: T4, e: T5; f: T6);

