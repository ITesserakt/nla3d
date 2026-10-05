// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "elements/element.h"

namespace nla3d {

class ElementTETRA1 : public ElementTETRA {
  public:
    ElementTETRA1();

    void pre() override;

    void buildK() override;

    void update() override;

    void makeB(math::Mat<3, 4>& B);

    void makeC(math::MatSym<3>& C) const;

    // conductivity coef ( W/(K m), for example)
    double k = 0.0;

    // flux[X], flux[Y], flux[Z]
    math::Vec<3> flux;

    double vol = 0.0;

    // postproc procedures
    bool getScalar(double* scalar, scalarQuery code, uint16 gp, double scale) override;

    bool getVector(math::Vec<3>& vector, vectorQuery code, uint16 gp, double scale) override;
};

} // namespace nla3d
