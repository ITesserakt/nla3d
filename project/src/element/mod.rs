use std::ops::IndexMut;

mod isoparametric;
mod quadrature;
mod vertex;

#[non_exhaustive]
pub enum ElementShape {
    Vertex = 0,
    TwinVertex,
    Line,
    Triangle,
    Quad,
    Tetra,
    Hexahedron,
    Wedge,
    Pyramid,
    QuadraticEdge,
    QuadraticTriangle,
    QuadraticQuad,
    QuadraticTetra,
    QuadraticHexahedron
}

impl ElementShape {
    #[inline]
    pub const fn dim(&self) -> u16 {
        match self {
            ElementShape::Vertex => 0,
            ElementShape::TwinVertex => 0,
            ElementShape::Line => 1,
            ElementShape::Triangle => 2,
            ElementShape::Quad => 2,
            ElementShape::Tetra => 3,
            ElementShape::Hexahedron => 3,
            ElementShape::Wedge => 3,
            ElementShape::Pyramid => 3,
            ElementShape::QuadraticEdge => 2,
            ElementShape::QuadraticTriangle => 2,
            ElementShape::QuadraticQuad => 2,
            ElementShape::QuadraticTetra => 3,
            ElementShape::QuadraticHexahedron => 3,
        }
    }
    
    #[inline]
    pub const fn num_nodes(&self) -> u16 {
        match self {
            ElementShape::Vertex => 1,
            ElementShape::TwinVertex => 2,
            ElementShape::Line => 2,
            ElementShape::Triangle => 3,
            ElementShape::Quad => 4,
            ElementShape::Tetra => 4,
            ElementShape::Hexahedron => 8,
            ElementShape::Wedge => 6,
            ElementShape::Pyramid => 5,
            ElementShape::QuadraticEdge => 3,
            ElementShape::QuadraticTriangle => 6,
            ElementShape::QuadraticQuad => 8,
            ElementShape::QuadraticTetra => 10,
            ElementShape::QuadraticHexahedron => 20
        }
    }
}

pub trait Element: IndexMut<usize> {
    const SHAPE: ElementShape;
}
