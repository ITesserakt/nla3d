// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "elements/element.h"

namespace nla3d {

class ElementINTER3 : public ElementWEDGE {
  public:
    ElementINTER3();

    void pre() override;

    void buildK() override;
    void make_D(Eigen::MatrixXd& D) const;
    Eigen::MatrixXd make_B(uint16 np, uint16 npj);
    Eigen::MatrixXd make_subB(uint16 np, uint16 npj);
    Eigen::MatrixXd make_T();

    void makeJacob();

    void update() override;

    uint16 getNNodes();
    uint16 getDim();

    uint16 nOfIntPoints();

    double intWeight(uint16 np) const;
    double intL1(uint16 np) const;
    double intL2(uint16 np) const;
    double intL3(uint16 np) const;

    double radoIntWeight(uint16 np, uint16 npj);
    double radoIntL1(uint16 np);
    double radoIntL2(uint16 np, uint16 npj);
    double radoIntL3(uint16 np, uint16 npj);

    // strength
    double kn = 0.0, ks = 0.0;

    // surface averaged strains
    math::Vec<3> strains;
    // surface averaged stress
    math::Vec<3> stress;
    // normal to triangle
    math::Vec<3> normal;

    uint16 i_int = 4; // index of integration scheme
    double det = 0.;  // determinant of Jacob matrix

    // postproc procedures
    bool getVector(math::Vec<3>& vector, vectorQuery query, uint16 gp, double scale) override;
    bool getTensor(math::MatSym<3>& tensor, tensorQuery query, uint16 gp, double scale) override;
};

inline uint16 ElementINTER3::getNNodes() { return 6; }

inline uint16 ElementINTER3::getDim() { return 3; }

// structures for convenient keeping of quadrature constants
struct TrianglePt {
    double L1;
    double L2;
    double L3;
    double W;
};

struct RadoPt {
    double AJ;
    double H;
    double AI;
    double AS;
};

// gauss quadrature for 3D trinangle from (0) to (1) in L-coordinates
// 1st order
static constexpr TrianglePt _triangle_o1[] = {
    {.L1 = 1. / 3., .L2 = 1. / 3., .L3 = 1. / 3., .W = 1.}
};

// 2nd order
static constexpr TrianglePt _triangle_o2[] = {
    {.L1 = 1. / 2.,      .L2 = 0., .L3 = 1. / 2., .W = 1. / 3.},
    {.L1 = 1. / 2., .L2 = 1. / 2.,      .L3 = 0., .W = 1. / 3.},
    {     .L1 = 0., .L2 = 1. / 2., .L3 = 1. / 2., .W = 1. / 3.}
};

// 3d order
static constexpr TrianglePt _triangle_o3[] = {
    {  .L1 = 1. / 3.,   .L2 = 1. / 3.,   .L3 = 1. / 3., .W = -27. / 48.},
    {.L1 = 11. / 15.,  .L2 = 2. / 15.,  .L3 = 2. / 15.,  .W = 25. / 48.},
    { .L1 = 2. / 15.,  .L2 = 2. / 15., .L3 = 11. / 15.,  .W = 25. / 48.},
    { .L1 = 2. / 15., .L2 = 11. / 15.,  .L3 = 2. / 15.,  .W = 25. / 48.}
};

// 4 order
static const TrianglePt _triangle_o4[] = {
    {.L1 = 1. / 3., .L2 = 1. / 3., .L3 = 1. / 3., .W = 27. / 60.},
    {.L1 = 1. / 2., .L2 = 1. / 2.,      .L3 = 0.,  .W = 8. / 60.},
    {     .L1 = 0., .L2 = 1. / 2., .L3 = 1. / 2.,  .W = 8. / 60.},
    {.L1 = 1. / 2.,      .L2 = 0., .L3 = 1. / 2.,  .W = 8. / 60.},
    {     .L1 = 1.,      .L2 = 0.,      .L3 = 0.,  .W = 3. / 60.},
    {     .L1 = 0.,      .L2 = 1.,      .L3 = 0.,  .W = 3. / 60.},
    {     .L1 = 0.,      .L2 = 0.,      .L3 = 1.,  .W = 3. / 60.}
};

// 5 order
static const TrianglePt _triangle_o5[] = {
    {   .L1 = 1. / 3.,    .L2 = 1. / 3.,    .L3 = 1. / 3.,      .W = 0.225},
    {.L1 = 0.05971587, .L2 = 0.47014206, .L3 = 0.47014206, .W = 0.13239415},
    {.L1 = 0.47014206, .L2 = 0.05971587, .L3 = 0.47014206, .W = 0.13239415},
    {.L1 = 0.47014206, .L2 = 0.47014206, .L3 = 0.05971587, .W = 0.13239415},
    {.L1 = 0.79742699, .L2 = 0.10128651, .L3 = 0.10128651, .W = 0.12593918},
    {.L1 = 0.10128651, .L2 = 0.79742699, .L3 = 0.10128651, .W = 0.12593918},
    {.L1 = 0.10128651, .L2 = 0.10128651, .L3 = 0.79742699, .W = 0.12593918}
};

static constexpr RadoPt _rado_o3[] = {
    {.AJ = 0.1127016654, .H = 0.2777777778, .AI = 0.0885879595, .AS = 0.2204622112},
    {         .AJ = 0.5, .H = 0.4444444444, .AI = 0.4094668644, .AS = 0.3881934688},
    {.AJ = 0.8872983346, .H = 0.2777777778, .AI = 0.7876594618, .AS = 0.3288443200}
};

// array of number of quadrature points in integration scheme
static constexpr uint16 _np_triangle[] = {
    sizeof(_triangle_o1) / sizeof(TrianglePt), sizeof(_triangle_o2) / sizeof(TrianglePt),
    sizeof(_triangle_o3) / sizeof(TrianglePt), sizeof(_triangle_o4) / sizeof(TrianglePt),
    sizeof(_triangle_o5) / sizeof(TrianglePt)};

inline uint16 ElementINTER3::nOfIntPoints() {
    // return _np_triangle[i_int];
    return 3;
}

static const TrianglePt* _table_triangle[] = {_triangle_o1, _triangle_o2, _triangle_o3, _triangle_o4, _triangle_o5};

inline double ElementINTER3::intWeight(const uint16 np) const { return _table_triangle[i_int][np].W * det; }

inline double ElementINTER3::radoIntWeight(const uint16 np, const uint16 npj) {
    return _rado_o3[np].AS * _rado_o3[npj].H * (1 - radoIntL1(np)) * det;
}

inline double ElementINTER3::intL1(const uint16 np) const { return _table_triangle[i_int][np].L1; }

inline double ElementINTER3::radoIntL1(const uint16 np) { return _rado_o3[np].AI; }

inline double ElementINTER3::intL2(const uint16 np) const { return _table_triangle[i_int][np].L2; }

inline double ElementINTER3::radoIntL2(const uint16 np, const uint16 npj) {
    return _rado_o3[npj].AJ * (1 - radoIntL1(np));
}

inline double ElementINTER3::intL3(const uint16 np) const { return _table_triangle[i_int][np].L3; }

inline double ElementINTER3::radoIntL3(const uint16 np, const uint16 npj) {
    return 1. - radoIntL1(np) - radoIntL2(np, npj);
}

} // namespace nla3d
