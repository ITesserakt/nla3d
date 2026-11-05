use core::marker::PhantomData;

use generic_array::typenum::Const;

use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind, zip::ElementwiseConstructible};

pub trait Layout {
    type Matrix<V>: ElementwiseConstructible<Item = V>;
}

pub trait Join<R: Layout>: Layout {
    type Out: Layout;
}

pub trait Fits<Rhs = Self>: Layout {}

pub struct Any;
pub struct Dynamic;
pub struct Static<const N: usize, const M: usize>;
pub struct Symmetric<const N: usize, Kind>(PhantomData<Kind>);

impl Layout for Any {
    type Matrix<V> = DMat<V>;
}

impl Layout for Dynamic {
    type Matrix<V> = DMat<V>;
}

impl<const N: usize, const M: usize> Layout for Static<N, M> {
    type Matrix<V> = Mat<N, M, V>;
}

impl<const N: usize, Kind> Layout for Symmetric<N, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
{
    type Matrix<V> = MatSym<N, V, Kind>;
}

impl<L: Layout> Join<Any> for L {
    type Out = L;
}

impl<L: Layout> Join<Dynamic> for L {
    type Out = Dynamic;
}

impl<const N: usize, const M: usize> Join<Static<N, M>> for Any {
    type Out = Static<N, M>;
}

impl<const N: usize, Kind> Join<Symmetric<N, Kind>> for Any
where
    Const<N>: MatSymLength,
    Kind: SymKind,
{
    type Out = Symmetric<N, Kind>;
}

impl<const N: usize, const M: usize> Join<Static<N, M>> for Dynamic {
    type Out = Dynamic;
}

impl<const N: usize, Kind: SymKind> Join<Symmetric<N, Kind>> for Dynamic
where
    Const<N>: MatSymLength,
{
    type Out = Dynamic;
}

impl<const N: usize, const M: usize> Join<Static<N, M>> for Static<N, M> {
    type Out = Static<N, M>;
}

impl<const N: usize, Kind: SymKind> Join<Symmetric<N, Kind>> for Symmetric<N, Kind>
where
    Const<N>: MatSymLength,
{
    type Out = Symmetric<N, Kind>;
}

impl<const N: usize, Kind: SymKind> Join<Symmetric<N, Kind>> for Static<N, N>
where
    Const<N>: MatSymLength,
{
    type Out = Static<N, N>;
}

impl<const N: usize, Kind: SymKind> Join<Static<N, N>> for Symmetric<N, Kind>
where
    Const<N>: MatSymLength,
{
    type Out = Static<N, N>;
}

impl<T: Layout> Fits<T> for Any {}

impl<const N: usize, const M: usize> Fits<Static<N, M>> for Static<N, M> {}
impl<const N: usize, const M: usize> Fits<Any> for Static<N, M> {}
impl<const N: usize, const M: usize> Fits<Dynamic> for Static<N, M> {}

impl<const N: usize, Kind: SymKind> Fits<Symmetric<N, Kind>> for Symmetric<N, Kind> where
    Const<N>: MatSymLength
{
}
impl<const N: usize, Kind: SymKind> Fits<Any> for Symmetric<N, Kind> where Const<N>: MatSymLength {}
impl<const N: usize, Kind: SymKind> Fits<Dynamic> for Symmetric<N, Kind> where Const<N>: MatSymLength
{}

impl<const N: usize, Kind: SymKind> Fits<Static<N, N>> for Symmetric<N, Kind> where
    Const<N>: MatSymLength
{
}
