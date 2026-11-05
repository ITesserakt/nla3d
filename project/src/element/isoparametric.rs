use crate::element::quadrature::{GaussQuadrature, TensorGaussQuadrature};

const DEFAULT_QUADRATURE_ITERATIONS: usize = 10;

static QUAD_LINE_O1: GaussQuadrature<1> = GaussQuadrature::new(DEFAULT_QUADRATURE_ITERATIONS);
static QUAD_LINE_O2: GaussQuadrature<2> = GaussQuadrature::new(DEFAULT_QUADRATURE_ITERATIONS);
static QUAD_LINE_O3: GaussQuadrature<3> = GaussQuadrature::new(DEFAULT_QUADRATURE_ITERATIONS);

static QUAD_QUAD_O1: TensorGaussQuadrature<2, 1> = TensorGaussQuadrature::new::<1>(DEFAULT_QUADRATURE_ITERATIONS);
static QUAD_QUAD_O2: TensorGaussQuadrature<2, 4> = TensorGaussQuadrature::new::<2>(DEFAULT_QUADRATURE_ITERATIONS);
static QUAD_QUAD_O3: TensorGaussQuadrature<2, 9> = TensorGaussQuadrature::new::<3>(DEFAULT_QUADRATURE_ITERATIONS);

static QUAD_HEXAHEDRON_O1: TensorGaussQuadrature<3, 1> = TensorGaussQuadrature::new::<1>(DEFAULT_QUADRATURE_ITERATIONS);
static QUAD_HEXAHEDRON_O2: TensorGaussQuadrature<3, 8> = TensorGaussQuadrature::new::<2>(DEFAULT_QUADRATURE_ITERATIONS);
static QUAD_HEXAHEDRON_O3: TensorGaussQuadrature<3, 27> = TensorGaussQuadrature::new::<3>(DEFAULT_QUADRATURE_ITERATIONS);
