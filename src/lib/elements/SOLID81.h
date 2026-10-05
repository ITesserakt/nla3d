// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "FEStorage.h"
#include "elements/element.h"
#include "elements/isoparametric.h"

namespace nla3d {

// 8-node hexahedron nonlinear element based on mixed approach
class ElementSOLID81 : public ElementIsoParamHEXAHEDRON {
  public:
    ElementSOLID81() {
        intOrder = 2;
        type = ElementType::SOLID81;
    }
    ElementSOLID81(const ElementSOLID81& from) : ElementIsoParamHEXAHEDRON(from) { operator=(from); }

    // solving procedures
    void pre() override;
    void buildK() override;
    void update() override;

    void make_B_L(uint16 nPoint, math::Mat<6, 24>& B);  // функция создает линейную матрицу [B]
    void make_B_NL(uint16 nPoint, math::Mat<9, 24>& B); // функция создает линейную матрицу [Bomega]
    void make_S(uint16 nPoint, math::MatSym<9>& B);
    void make_Omega(uint16 nPoint, math::Mat<6, 9>& B);

    // postproc procedures
    bool getScalar(double* scalar, scalarQuery code, uint16 gp, double scale) override;
    bool getVector(math::Vec<3>& vector, vectorQuery code, uint16 gp, double scale) override;
    bool getTensor(math::MatSym<3>& tensor, tensorQuery code, uint16 gp, double scale) override;

    // internal element data
    // S[M_XX], S[M_XY], S[M_XZ], S[M_YY], S[M_YZ], S[M_ZZ]
    std::vector<math::Vec<6>> S; // S[номер т. интегр.][номер напряжения] - напряжения Пиолы-Кирхгоффа
    // C[M_XX], C[M_XY], C[M_XZ], C[M_YY], C[M_YZ], C[M_ZZ]
    std::vector<math::Vec<6>> C; // C[номер т. интегр.][номер деформ.] - компоненты тензора меры деформации
    // O[0]-dU/dx	O[1]-dU/dy	O[2]-dU/dz	O[3]-dV/dx	O[4]-dV/dy	O[5]-dV/dz	O[6]-dW/dx
    // O[7]-dW/dy	O[8]-dW/dz
    std::vector<math::Vec<9>> O; // S[номер т. интегр.][номер омеги]

    template <uint16 dimM, uint16 dimN>
    void assemble2(math::MatSym<dimM>& Kuu, math::Mat<dimM, dimM>& Kup, math::Mat<dimN, dimN>& Kpp, math::Vec<dimM>& Fu,
                   math::Vec<dimN>& Fp);
    template <uint16 dimM>
    void assemble3(math::MatSym<dimM>& Kuu, math::Vec<dimM>& Kup, double Kpp, math::Vec<dimM>& Fu, double Fp);
};

template <uint16 dimM>
void ElementSOLID81::assemble3(math::MatSym<dimM>& Kuu, math::Vec<dimM>& Kup, const double Kpp, math::Vec<dimM>& Fu,
                               const double Fp) {
    constexpr uint16 dim = 3;
    assert(getNNodes() * dim == dimM);
    const double* Kuu_p = Kuu.ptr();
    const double* Kup_p = Kup.ptr();
    const double* Fu_p = Fu.ptr();
    const Dof::dofType dofVec[] = {Dof::UX, Dof::UY, Dof::UZ};
    for (uint16 i = 0; i < getNNodes(); i++)
        for (uint16 di = 0; di < dim; di++)
            for (uint16 j = i; j < getNNodes(); j++)

                for (uint16 dj = 0; dj < dim; dj++) {
                    if ((i == j) && (dj < di))
                        continue;
                    storage->addValueK(nodes[i], dofVec[di], nodes[j], dofVec[dj], *Kuu_p);
                    ++Kuu_p;
                }
    // upper diagonal process for nodes-el dofs
    const uint32 elEq = storage->getElementDofEqNumber(getElNum(), Dof::HYDRO_PRESSURE);
    for (uint16 i = 0; i < getNNodes(); i++) {
        for (const auto di : dofVec) {
            const uint32 rowEq = storage->getNodeDofEqNumber(nodes[i], di);
            storage->addValueK(rowEq, elEq, *Kup_p);
            ++Kup_p;
        }
    }
    // upper diagonal process for el-el dofs
    storage->addValueK(elEq, elEq, Kpp);

    for (uint16 i = 0; i < getNNodes(); i++) {
        for (const auto di : dofVec) {
            storage->addValueF(nodes[i], di, *Fu_p);
            ++Fu_p;
        }
    }
    storage->addValueF(elEq, Fp);
}

} // namespace nla3d
