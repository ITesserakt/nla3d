use std::ops::{Index, IndexMut};
use crate::element::{Element, ElementShape};

pub struct Vertex {
    nodes: [u32; 1]
}

impl Element for Vertex {
    const SHAPE: ElementShape = ElementShape::Vertex;
}

impl Index<usize> for Vertex {
    type Output = u32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl IndexMut<usize> for Vertex {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}