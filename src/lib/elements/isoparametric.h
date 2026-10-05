// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "elements/element.h"
#include "sys.h"

// Formulations for isoparametric FE for different shapes. Isoparametic formulation means usage of the
// same shape functions for geometry interpolation and for field (displacement, for instance)
//   interpolation.
// These derived classes specify next things:
// 1. Specify form functions
// 2. Specify integration rule (Gauss quadrature)
// 3. Specify field derivatives in local and global coordinates

namespace nla3d {

// structures for convenient keeping of quadrature constants
struct QuadPt1D {
    double r;
    double w;
};

struct QuadPt2D {
    double r;
    double s;
    double w;
};

struct QuadPt3D {
    double r;
    double s;
    double t;
    double w;
};

// gauss quadrature for 1D line from (-1) to (1)
// 1st order
static constexpr QuadPt1D _quad_line_o1[] = {
    {.r = +0.0000000000000000, .w = +2.0000000000000000}
};

// 2nd order
static constexpr QuadPt1D _quad_line_o2[] = {
    {.r = -0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .w = +1.0000000000000000}
};

// 3rd order
static constexpr QuadPt1D _quad_line_o3[] = {
    {.r = -0.7745966692414834, .w = +0.5555555555555556},
    {.r = +0.0000000000000000, .w = +0.8888888888888888},
    {.r = +0.7745966692414834, .w = +0.5555555555555556}
};

// array of number of quadrature points in integration scheme
static constexpr uint16 _np_line[] = {sizeof(_quad_line_o1) / sizeof(QuadPt1D),
                                      sizeof(_quad_line_o2) / sizeof(QuadPt1D),
                                      sizeof(_quad_line_o3) / sizeof(QuadPt1D)};

static const QuadPt1D* _table_line[] = {_quad_line_o1, _quad_line_o2, _quad_line_o3};

// gauss quadrature for 2D rect from (-1,-1) to (1,1)
// 1st order
static constexpr QuadPt2D _quad_quad_o1[] = {
    {.r = +0.0000000000000000, .s = +0.0000000000000000, .w = +4.0000000000000000}
};

// 2nd order
static constexpr QuadPt2D _quad_quad_o2[] = {
    {.r = -0.5773502691896258, .s = -0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .s = -0.5773502691896258, .w = +1.0000000000000000},
    {.r = -0.5773502691896258, .s = +0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .s = +0.5773502691896258, .w = +1.0000000000000000}
};

// 3rd order
static const QuadPt2D _quad_quad_o3[] = {
    {.r = -0.7745966692414834, .s = -0.7745966692414834, .w = +0.3086419753086420},
    {.r = +0.0000000000000000, .s = -0.7745966692414834, .w = +0.4938271604938271},
    {.r = +0.7745966692414834, .s = -0.7745966692414834, .w = +0.3086419753086420},

    {.r = -0.7745966692414834, .s = +0.0000000000000000, .w = +0.4938271604938271},
    {.r = +0.0000000000000000, .s = +0.0000000000000000, .w = +0.7901234567901234},
    {.r = +0.7745966692414834, .s = +0.0000000000000000, .w = +0.4938271604938271},

    {.r = -0.7745966692414834, .s = +0.7745966692414834, .w = +0.3086419753086420},
    {.r = +0.0000000000000000, .s = +0.7745966692414834, .w = +0.4938271604938271},
    {.r = +0.7745966692414834, .s = +0.7745966692414834, .w = +0.3086419753086420}
};

// array of number of quadrature points in integration scheme
static constexpr uint16 _np_quad[] = {sizeof(_quad_quad_o1) / sizeof(QuadPt2D),
                                      sizeof(_quad_quad_o2) / sizeof(QuadPt2D),
                                      sizeof(_quad_quad_o3) / sizeof(QuadPt2D)};

static const QuadPt2D* _table_quad[] = {_quad_quad_o1, _quad_quad_o2, _quad_quad_o3};

// gauss quadrature for 3D brick from (-1,-1,-1) to (1,1,1)
// 1st order
static constexpr QuadPt3D _quad_hexahedron_o1[] = {
    {.r = +0.0000000000000000, .s = +0.0000000000000000, .t = +0.0000000000000000, .w = +8.0000000000000000}
};

// 2nd order
static const QuadPt3D _quad_hexahedron_o2[] = {
    {.r = -0.5773502691896258, .s = -0.5773502691896258, .t = -0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .s = -0.5773502691896258, .t = -0.5773502691896258, .w = +1.0000000000000000},
    {.r = -0.5773502691896258, .s = +0.5773502691896258, .t = -0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .s = +0.5773502691896258, .t = -0.5773502691896258, .w = +1.0000000000000000},

    {.r = -0.5773502691896258, .s = -0.5773502691896258, .t = +0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .s = -0.5773502691896258, .t = +0.5773502691896258, .w = +1.0000000000000000},
    {.r = -0.5773502691896258, .s = +0.5773502691896258, .t = +0.5773502691896258, .w = +1.0000000000000000},
    {.r = +0.5773502691896258, .s = +0.5773502691896258, .t = +0.5773502691896258, .w = +1.0000000000000000}
};

// 3rd order
static const QuadPt3D _quad_hexahedron_o3[] = {
    {.r = -0.7745966692414834, .s = -0.7745966692414834, .t = -0.7745966692414834, .w = +0.1714677640603567},
    {.r = +0.0000000000000000, .s = -0.7745966692414834, .t = -0.7745966692414834, .w = +0.2743484224965707},
    {.r = +0.7745966692414834, .s = -0.7745966692414834, .t = -0.7745966692414834, .w = +0.1714677640603567},

    {.r = -0.7745966692414834, .s = +0.0000000000000000, .t = -0.7745966692414834, .w = +0.2743484224965707},
    {.r = +0.0000000000000000, .s = +0.0000000000000000, .t = -0.7745966692414834, .w = +0.4389574759945130},
    {.r = +0.7745966692414834, .s = +0.0000000000000000, .t = -0.7745966692414834, .w = +0.2743484224965707},

    {.r = -0.7745966692414834, .s = +0.7745966692414834, .t = -0.7745966692414834, .w = +0.1714677640603567},
    {.r = +0.0000000000000000, .s = +0.7745966692414834, .t = -0.7745966692414834, .w = +0.2743484224965707},
    {.r = +0.7745966692414834, .s = +0.7745966692414834, .t = -0.7745966692414834, .w = +0.1714677640603567},

    {.r = -0.7745966692414834, .s = -0.7745966692414834, .t = +0.0000000000000000, .w = +0.2743484224965707},
    {.r = +0.0000000000000000, .s = -0.7745966692414834, .t = +0.0000000000000000, .w = +0.4389574759945130},
    {.r = +0.7745966692414834, .s = -0.7745966692414834, .t = +0.0000000000000000, .w = +0.2743484224965707},

    {.r = -0.7745966692414834, .s = +0.0000000000000000, .t = +0.0000000000000000, .w = +0.4389574759945130},
    {.r = +0.0000000000000000, .s = +0.0000000000000000, .t = +0.0000000000000000, .w = +0.7023319615912207},
    {.r = +0.7745966692414834, .s = +0.0000000000000000, .t = +0.0000000000000000, .w = +0.4389574759945130},

    {.r = -0.7745966692414834, .s = +0.7745966692414834, .t = +0.0000000000000000, .w = +0.2743484224965707},
    {.r = +0.0000000000000000, .s = +0.7745966692414834, .t = +0.0000000000000000, .w = +0.4389574759945130},
    {.r = +0.7745966692414834, .s = +0.7745966692414834, .t = +0.0000000000000000, .w = +0.2743484224965707},

    {.r = -0.7745966692414834, .s = -0.7745966692414834, .t = +0.7745966692414834, .w = +0.1714677640603567},
    {.r = +0.0000000000000000, .s = -0.7745966692414834, .t = +0.7745966692414834, .w = +0.2743484224965707},
    {.r = +0.7745966692414834, .s = -0.7745966692414834, .t = +0.7745966692414834, .w = +0.1714677640603567},

    {.r = -0.7745966692414834, .s = +0.0000000000000000, .t = +0.7745966692414834, .w = +0.2743484224965707},
    {.r = +0.0000000000000000, .s = +0.0000000000000000, .t = +0.7745966692414834, .w = +0.4389574759945130},
    {.r = +0.7745966692414834, .s = +0.0000000000000000, .t = +0.7745966692414834, .w = +0.2743484224965707},

    {.r = -0.7745966692414834, .s = +0.7745966692414834, .t = +0.7745966692414834, .w = +0.1714677640603567},
    {.r = +0.0000000000000000, .s = +0.7745966692414834, .t = +0.7745966692414834, .w = +0.2743484224965707},
    {.r = +0.7745966692414834, .s = +0.7745966692414834, .t = +0.7745966692414834, .w = +0.1714677640603567}
};

static constexpr uint16 _np_hexahedron[] = {sizeof(_quad_hexahedron_o1) / sizeof(QuadPt3D),
                                            sizeof(_quad_hexahedron_o2) / sizeof(QuadPt3D),
                                            sizeof(_quad_hexahedron_o3) / sizeof(QuadPt3D)};

static const QuadPt3D* _table_hexahedron[] = {_quad_hexahedron_o1, _quad_hexahedron_o2, _quad_hexahedron_o3};

class ElementIsoParamLINE : public ElementLINE {
  public:
    double det; // Jacobian

    void makeJacob();

    double intWeight(uint16 np) const;
    double volume() const;
    void np2rst(uint16 np, double* xi) const; // by number of gauss point find local coordinates
    uint16 nOfIntPoints() const;

    // get form function values in local point (r, s)
    math::Vec<2> formFunc(double r);
    // get form function values in integration point np
    math::Vec<2> formFunc(uint16 np);
    math::Mat<2, 1> formFuncDeriv(double r);

  protected:
    uint16 i_int = 0; // index of integration scheme
};

class ElementIsoParamQUAD : public ElementQUAD {
  public:
    std::vector<math::Mat<4, 2>> NiXj; // derivates form function / local coordinates
    std::vector<double> det;           // Jacobian

    double sideDet[4]; // Jacobian for side integration

    // function to calculate all staff for isoparametric FE
    void makeJacob();
    double intWeight(uint16 np) const;
    double volume() const;
    void np2rst(uint16 np, double* xi) const; // by number of gauss point find local coordinates
    uint16 nOfIntPoints() const;

    // get form function values in local point (r, s)
    math::Vec<4> formFunc(double r, double s);
    // get form function values in integration point np
    math::Vec<4> formFunc(uint16 np);
    math::Mat<4, 2> formFuncDeriv(double r, double s);

  protected:
    uint16 i_int = 0; // index of integration scheme
};

class ElementIsoParamHEXAHEDRON : public ElementHEXAHEDRON {
  public:
    std::vector<math::Mat<8, 3>> NiXj; // derivates form function / local coordinates
    std::vector<double> det;           // Jacobian

    // function to calculate all staff for isoparametric FE
    void makeJacob();
    double intWeight(uint16 np) const;
    double volume() const;
    void np2rst(uint16 np, double* xi) const; // by number of gauss point find local coordinates
    uint16 nOfIntPoints() const;
    // get form function values in local point (r, s, t)
    math::Vec<8> formFunc(double r, double s, double t);
    // get form function values in integration point np
    math::Vec<8> formFunc(uint16 np);
    // get form function derivatives (vs. r,s,t) in local point (r,s,t)
    math::Mat<8, 3> formFuncDeriv(double r, double s, double t);

  protected:
    uint16 i_int = 0; // index of integration scheme
};

inline double ElementIsoParamLINE::intWeight(const uint16 np) const {
    assert(np < _np_line[i_int]);
    assert(det > 0.0);

    return _table_line[i_int][np].w * det;
}

inline void ElementIsoParamLINE::np2rst(const uint16 np, double* v) const {
    assert(np < _np_line[i_int]);
    v[0] = _table_line[i_int][np].r;
}

inline uint16 ElementIsoParamLINE::nOfIntPoints() const { return _np_line[i_int]; }

inline double ElementIsoParamQUAD::intWeight(const uint16 np) const {
    assert(np < _np_quad[i_int]);
    assert(!det.empty());

    return _table_quad[i_int][np].w * det[np];
}

inline void ElementIsoParamQUAD::np2rst(const uint16 np, double* v) const {
    assert(np < _np_quad[i_int]);
    v[0] = _table_quad[i_int][np].r;
    v[1] = _table_quad[i_int][np].s;
}

inline uint16 ElementIsoParamQUAD::nOfIntPoints() const { return _np_quad[i_int]; }

inline double ElementIsoParamHEXAHEDRON::intWeight(const uint16 np) const {
    assert(np < _np_hexahedron[i_int]);
    assert(!det.empty());

    return _table_hexahedron[i_int][np].w * det[np];
}

inline void ElementIsoParamHEXAHEDRON::np2rst(const uint16 np, double* v) const {
    assert(np < _np_hexahedron[i_int]);
    v[0] = _table_hexahedron[i_int][np].r;
    v[1] = _table_hexahedron[i_int][np].s;
    v[2] = _table_hexahedron[i_int][np].t;
}

inline uint16 ElementIsoParamHEXAHEDRON::nOfIntPoints() const { return _np_hexahedron[i_int]; }

} // namespace nla3d
