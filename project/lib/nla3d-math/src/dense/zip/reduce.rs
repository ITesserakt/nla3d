use faer::{MatMut, MatRef, Shape};
use generic_array::typenum::Const;

use super::construct::{ElementwiseConstructible, Nothing};
use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind, zip::producer::{Consume, Fill}};

pub trait Zippable<Rhs = Self> {}
impl<T> Zippable for T {}
impl<const N: usize, T, V, Kind> Zippable<Mat<N, N, V>> for MatSym<N, T, Kind> where
    Const<N>: MatSymLength
{
}
impl<const N: usize, T, V, Kind> Zippable<MatSym<N, V, Kind>> for Mat<N, N, T> where
    Const<N>: MatSymLength
{
}
impl<const N: usize, T, V, Kind, Rows: Shape, Cols: Shape> Zippable<MatSym<N, V, Kind>>
    for DMat<T, Rows, Cols>
where
    Const<N>: MatSymLength,
{
}
impl<const N: usize, T, V, Kind, Rows: Shape, Cols: Shape> Zippable<DMat<T, Rows, Cols>>
    for MatSym<N, V, Kind>
where
    Const<N>: MatSymLength,
{
}
impl<const N: usize, const M: usize, T, V, Rows: Shape, Cols: Shape> Zippable<Mat<N, M, T>>
    for DMat<V, Rows, Cols>
{
}
impl<const N: usize, const M: usize, T, V, Rows: Shape, Cols: Shape> Zippable<DMat<T, Rows, Cols>>
    for Mat<N, M, V>
{
}

pub trait RemoveRef {
    type Output;
}

impl<T> RemoveRef for &T {
    type Output = T;
}

impl<T> RemoveRef for &mut T {
    type Output = T;
}

impl<T, Rows: Shape, Cols: Shape, RStrides, CStrides> RemoveRef
    for MatRef<'_, T, Rows, Cols, RStrides, CStrides>
{
    type Output = faer::Mat<T, Rows, Cols>;
}

impl<T, Rows: Shape, Cols: Shape, RStrides, CStrides> RemoveRef
    for MatMut<'_, T, Rows, Cols, RStrides, CStrides>
{
    type Output = faer::Mat<T, Rows, Cols>;
}

impl<T> RemoveRef for Consume<T> {
    type Output = T;
}

impl<T> RemoveRef for Fill<T> {
    type Output = Self;
}

pub trait ReduceToCommonMatrix<Rhs> {
    type Output<Item>;
}

pub trait ReducedToCommonMatrix<Item> {
    type Output: ElementwiseConstructible<Item = Item>;
}

impl<const N: usize, const M: usize, T, V> ReducedToCommonMatrix<V> for Mat<N, M, T> {
    type Output = Mat<N, M, V>;
}

impl<const N: usize, T, V, Kind> ReducedToCommonMatrix<V> for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
{
    type Output = MatSym<N, V, Kind>;
}

impl<T> ReducedToCommonMatrix<T> for () {
    type Output = Nothing<T>;
}

impl<T: ReduceToCommonMatrix<T>> ReduceToCommonMatrix<T> for ! {
    type Output<Item> = T::Output<Item>;
}

impl<const N: usize, T, V, Kind> ReduceToCommonMatrix<Mat<N, N, V>> for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
{
    type Output<Item> = Mat<N, N, Item>;
}

impl<const N: usize, const M: usize, T, V> ReduceToCommonMatrix<Mat<N, M, V>> for Mat<N, M, T> {
    type Output<Item> = Mat<N, M, Item>;
}

impl<const N: usize, T, V, Kind> ReduceToCommonMatrix<MatSym<N, V, Kind>> for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
{
    type Output<Item> = MatSym<N, Item, Kind>;
}

impl<const N: usize, T, V, Kind> ReduceToCommonMatrix<MatSym<N, V, Kind>> for Mat<N, N, T>
where
    Const<N>: MatSymLength,
{
    type Output<Item> = Mat<N, N, Item>;
}

impl<T, V, Rows: Shape, Cols: Shape> ReduceToCommonMatrix<DMat<T, Rows, Cols>>
    for DMat<V, Rows, Cols>
{
    type Output<Item> = DMat<Item, Rows, Cols>;
}

impl<const N: usize, Kind, T, V, Rows: Shape, Cols: Shape> ReduceToCommonMatrix<MatSym<N, T, Kind>>
    for DMat<V, Rows, Cols>
where
    Const<N>: MatSymLength,
{
    type Output<Item> = DMat<Item, Rows, Cols>;
}

impl<const N: usize, const M: usize, T, V, Rows: Shape, Cols: Shape>
    ReduceToCommonMatrix<Mat<N, M, T>> for DMat<V, Rows, Cols>
{
    type Output<Item> = DMat<Item, Rows, Cols>;
}

impl<const N: usize, const M: usize, T, V, Rows: Shape, Cols: Shape>
    ReduceToCommonMatrix<DMat<T, Rows, Cols>> for Mat<N, M, V>
{
    type Output<Item> = DMat<Item, Rows, Cols>;
}

impl<const N: usize, Kind, T, V, Rows: Shape, Cols: Shape> ReduceToCommonMatrix<DMat<V, Rows, Cols>>
    for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
{
    type Output<Item> = DMat<Item, Rows, Cols>;
}
