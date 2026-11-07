#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Free;
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Dynamic {
    pub rows: usize,
    pub cols: usize
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Static<const N: usize, const M: usize>;

pub trait Shape {}
pub trait KnownShape: Shape + Copy {
    fn materialize(self) -> (usize, usize);
}
pub trait Join<Rhs = Self>: Shape {
    type Output: Shape;

    fn join(self, rhs: Rhs) -> Self::Output;
}
pub trait Fits<Target = Self>: Shape {}

impl<S: Shape> Join<Free> for S {
    type Output = S;

    fn join(self, _: Free) -> Self::Output {
        self
    }
}

impl<const N: usize, const M: usize> Join<Static<N, M>> for Dynamic {
    type Output = Dynamic;

    fn join(self, _: Static<N, M>) -> Self::Output {
        self
    }
}

impl Join for Dynamic {
    type Output = Dynamic;

    fn join(self, rhs: Self) -> Self::Output {
        assert_eq!(self, rhs);
        rhs
    }
}

impl<const N: usize, const M: usize> Join<Static<N, M>> for Static<N, M> {
    type Output = Static<N, M>;

    fn join(self, rhs: Static<N, M>) -> Self::Output {
        rhs
    }
}

impl<const N: usize, const M: usize> Join<Dynamic> for Static<N, M> {
    type Output = Dynamic;

    fn join(self, rhs: Dynamic) -> Self::Output {
        assert_eq!(rhs.rows, N);
        assert_eq!(rhs.cols, M);

        rhs
    }
}

impl Join<Dynamic> for Free {
    type Output = Dynamic;

    fn join(self, rhs: Dynamic) -> Self::Output {
        rhs
    }
}

impl<const N: usize, const M: usize> Join<Static<N, M>> for Free {
    type Output = Static<N, M>;

    fn join(self, rhs: Static<N, M>) -> Self::Output {
        rhs
    }
}

impl Shape for Free {}
impl Shape for Dynamic {}
impl<const N: usize, const M: usize> Shape for Static<N, M> {}
impl KnownShape for Dynamic {
    fn materialize(self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}
impl<const N: usize, const M: usize> KnownShape for Static<N, M> {
    fn materialize(self) -> (usize, usize) {
        (N, M)
    }
}

impl<S: Shape> Fits<Dynamic> for S {}
impl<const N: usize, const M: usize> Fits for Static<N, M> {}
impl<const N: usize, const M: usize> Fits<Static<N, M>> for Free {}

const _: () = {
    const fn check_symmetry<A: Shape, B: Shape>()
    where
        A: Join<B>,
        B: Join<A, Output = A::Output>,
    {
    }

    check_symmetry::<Free, Free>();
    check_symmetry::<Dynamic, Dynamic>();
    check_symmetry::<Static<0, 0>, Static<0, 0>>();

    check_symmetry::<Free, Dynamic>();
    check_symmetry::<Free, Static<0, 0>>();

    check_symmetry::<Dynamic, Static<0, 0>>();
};
