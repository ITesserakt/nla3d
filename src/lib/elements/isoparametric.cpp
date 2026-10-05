
// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#include "isoparametric.h"
#include "elements/element.h"
#include "Node.h"

namespace nla3d {

void ElementIsoParamLINE::makeJacob() {
    // bound user-provided intOrder
    intOrder = (intOrder < 1) ? 1 : intOrder;
    intOrder = (intOrder > 3) ? 3 : intOrder;
    // we store integration scheme index in i_int
    // thus last change of intOrder will not affect the solution procedure
    i_int = intOrder - 1;

    const math::Vec<3> n1pos = storage->getNode(getNodeNumber(0)).pos;
    const math::Vec<3> n2pos = storage->getNode(getNodeNumber(1)).pos;
    det = (n2pos - n1pos).length() * 0.5;
}

double ElementIsoParamLINE::volume() const {
    double volume = 0.0;
    // NOTE: actually straight line can be integrated in constant time
    for (uint16 np = 0; np < _np_line[i_int]; np++)
        volume += intWeight(np);
    return volume;
}

math::Vec<2> ElementIsoParamLINE::formFunc(const double r) { return math::Vec<2>{0.5 * (1.0 - r), 0.5 * (1.0 + r)}; }

math::Vec<2> ElementIsoParamLINE::formFunc(const uint16 np) { return formFunc(_table_line[i_int][np].r); }

math::Mat<2, 1> ElementIsoParamLINE::formFuncDeriv(double) { return math::Mat<2, 1>{-0.5, 0.5}; }

void ElementIsoParamQUAD::makeJacob() {
    constexpr uint16 dim = 2;
    constexpr uint16 nodes_num = 4;

    // bound user-provided intOrder
    intOrder = (intOrder < 1) ? 1 : intOrder;
    intOrder = (intOrder > 3) ? 3 : intOrder;
    // we store integration scheme index in i_int
    // thus last change of intOrder will not affect the solution procedure
    i_int = intOrder - 1;

    // Jacob inv matrix

    det.clear();
    det.assign(_np_quad[i_int], 0.0);

    NiXj.clear();
    NiXj.resize(_np_quad[i_int]);

    math::Mat<nodes_num, dim> dN; // form function derivatives
    math::Mat<dim, dim> J;

    for (uint16 np = 0; np < _np_quad[i_int]; np++) {
        const QuadPt2D q = _table_quad[i_int][np];
        dN = formFuncDeriv(q.r, q.s);

        J.zero();

        for (uint16 nod = 0; nod < nodes_num; nod++) {
            math::Vec<3> pos = storage->getNode(getNodeNumber(nod)).pos;
            for (uint16 i = 0; i < dim; i++)
                for (uint16 j = 0; j < dim; j++)
                    J[i][j] += dN[nod][i] * pos[j];
        }

        det[np] = J.det(); // determinant of Jacob matrix
        // check for geometry form error
        LOG_IF(det[np] < 1.0e-20, ERROR) << "Determinant is too small (" << det[np] << ") in element = " << elNum;
        // обращение матрицы Якоби
        math::Mat<dim, dim> Jacob = J.inv(det[np]);

        // производные функций формы по глоб. координатам
        for (uint16 nod = 0; nod < nodes_num; nod++)
            for (uint16 i = 0; i < dim; i++)
                for (uint16 j = 0; j < dim; j++)
                    NiXj[np][nod][i] += Jacob[i][j] * dN[nod][j];
    }
}

math::Vec<4> ElementIsoParamQUAD::formFunc(const double r, const double s) {
    return math::Vec<4>{0.25 * (1.0 - r) * (1.0 - s), 0.25 * (1.0 + r) * (1.0 - s), 0.25 * (1.0 + r) * (1.0 + s),
                        0.25 * (1.0 - r) * (1.0 + s)};
}

math::Vec<4> ElementIsoParamQUAD::formFunc(const uint16 np) {
    const QuadPt2D q = _table_quad[i_int][np];
    return formFunc(q.r, q.s);
}

math::Mat<4, 2> ElementIsoParamQUAD::formFuncDeriv(const double r, const double s) {
    return math::Mat<4, 2>{-0.25 * (1.0 - s), -0.25 * (1.0 - r), +0.25 * (1.0 - s), -0.25 * (1.0 + r),
                           +0.25 * (1.0 + s), +0.25 * (1.0 + r), -0.25 * (1.0 + s), +0.25 * (1.0 - r)};
}

double ElementIsoParamQUAD::volume() const {
    double volume = 0.0;
    for (uint16 np = 0; np < _np_quad[i_int]; np++)
        volume += intWeight(np);
    return volume;
}

void ElementIsoParamHEXAHEDRON::makeJacob() {
    constexpr uint16 dim = 3;
    constexpr uint16 nodes_num = 8;

    // bound user-provided intOrder
    intOrder = (intOrder < 1) ? 1 : intOrder;
    intOrder = (intOrder > 3) ? 3 : intOrder;
    i_int = intOrder - 1;

    // Jacob inv matrix

    det.clear();
    det.assign(_np_hexahedron[i_int], 0.0);

    NiXj.clear();
    NiXj.resize(_np_hexahedron[i_int]);

    math::Mat<nodes_num, dim> dN; // form function derivatives
    math::Mat<dim, dim> J;

    for (uint16 np = 0; np < _np_hexahedron[i_int]; np++) {
        const QuadPt3D q = _table_hexahedron[i_int][np];
        dN = formFuncDeriv(q.r, q.s, q.t);

        J.zero();

        for (uint16 nod = 0; nod < nodes_num; nod++) {
            math::Vec<3> pos = storage->getNode(getNodeNumber(nod)).pos;
            for (uint16 i = 0; i < dim; i++)
                for (uint16 j = 0; j < dim; j++)
                    J[i][j] += dN[nod][i] * pos[j];
        }

        det[np] = J.det(); // determinant of Jacob matrix
        // check for geometry form error
        LOG_IF(det[np] < 1.0e-20, ERROR) << "Determinant is too small (" << det[np] << ") in element = " << elNum;
        // обращение матрицы Якоби
        math::Mat<dim, dim> Jacob = J.inv(det[np]);

        // производные функций формы по глоб. координатам
        for (uint16 nod = 0; nod < nodes_num; nod++)
            for (uint16 i = 0; i < dim; i++)
                for (uint16 j = 0; j < dim; j++)
                    NiXj[np][nod][i] += Jacob[i][j] * dN[nod][j];
    }
}

math::Vec<8> ElementIsoParamHEXAHEDRON::formFunc(const double r, const double s, const double t) {
    return math::Vec<8>{0.125 * (1.0 - r) * (1.0 - s) * (1.0 - t), 0.125 * (1.0 + r) * (1.0 - s) * (1.0 - t),
                        0.125 * (1.0 + r) * (1.0 + s) * (1.0 - t), 0.125 * (1.0 - r) * (1.0 + s) * (1.0 - t),

                        0.125 * (1.0 - r) * (1.0 - s) * (1.0 + t),

                        0.125 * (1.0 + r) * (1.0 - s) * (1.0 + t), 0.125 * (1.0 + r) * (1.0 + s) * (1.0 + t),
                        0.125 * (1.0 - r) * (1.0 + s) * (1.0 + t)};
}

math::Vec<8> ElementIsoParamHEXAHEDRON::formFunc(const uint16 np) {
    const QuadPt3D q = _table_hexahedron[i_int][np];
    return formFunc(q.r, q.s, q.t);
}

math::Mat<8, 3> ElementIsoParamHEXAHEDRON::formFuncDeriv(const double r, const double s, const double t) {
    return math::Mat<8, 3>{
        -0.125 * (1.0 - s) * (1.0 - t), -0.125 * (1.0 - r) * (1.0 - t), -0.125 * (1.0 - r) * (1.0 - s),
        +0.125 * (1.0 - s) * (1.0 - t), -0.125 * (1.0 + r) * (1.0 - t), -0.125 * (1.0 + r) * (1.0 - s),
        +0.125 * (1.0 + s) * (1.0 - t), +0.125 * (1.0 + r) * (1.0 - t), -0.125 * (1.0 + r) * (1.0 + s),
        -0.125 * (1.0 + s) * (1.0 - t), +0.125 * (1.0 - r) * (1.0 - t), -0.125 * (1.0 - r) * (1.0 + s),

        -0.125 * (1.0 - s) * (1.0 + t), -0.125 * (1.0 - r) * (1.0 + t), +0.125 * (1.0 - r) * (1.0 - s),
        +0.125 * (1.0 - s) * (1.0 + t), -0.125 * (1.0 + r) * (1.0 + t), +0.125 * (1.0 + r) * (1.0 - s),
        +0.125 * (1.0 + s) * (1.0 + t), +0.125 * (1.0 + r) * (1.0 + t), +0.125 * (1.0 + r) * (1.0 + s),
        -0.125 * (1.0 + s) * (1.0 + t), +0.125 * (1.0 - r) * (1.0 + t), +0.125 * (1.0 - r) * (1.0 + s)};
}

double ElementIsoParamHEXAHEDRON::volume() const {
    double volume = 0.0;
    for (uint16 np = 0; np < _np_hexahedron[i_int]; np++)
        volume += intWeight(np);
    return volume;
}

} // namespace nla3d
