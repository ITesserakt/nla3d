// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "sys.h"
#include <Eigen/Dense>
#include <initializer_list>
#include <vector>

#include "Dof.h"
#include "math/Mat.h"
#include "math/Vec.h"
#include "query.h"

namespace nla3d {

// pre-defines
class Material;
class FEStorage;

// Element shapes are taken from vtk-file-formats.pdf
enum ElementShape {
    VERTEX = 0,
    TWIN_VERTEX,
    LINE,
    TRIANGLE,
    QUAD,
    TETRA,
    HEXAHEDRON,
    WEDGE,
    PYRAMID,
    QUADRARIC_EDGE,
    QUADRATIC_TRIANGLE,
    QUADRATIC_QUAD,
    QUADRATIC_TETRA,
    QUADRARIC_HEXAHEDRON,
    UNDEFINED
};

// number of dimensions in shape
static const uint16 _shape_dim[] = {
    0, // VERTEX
    0, // TWIN_VERTEX
    1, // LINE
    2, // TRIANGLE
    2, // QUAD
    3, // TETRA
    3, // HEXAHEDRON
    3, // WEDGE
    3, // PYRAMID
    2, // QUADRARIC_EDGE
    2, // QUADRATIC_TRIANGLE
    2, // QUADRATIC_QUAD
    3, // QUADRATIC_TETRA
    3  // QUADRARIC_HEXAHEDRON
};

// number of nodes in shape
static const uint16 _shape_nnodes[] = {
    1,  // VERTEX,
    2,  // TWIN_VERTEX
    2,  // LINE
    3,  // TRIANGLE
    4,  // QUAD
    4,  // TETRA
    8,  // HEXAHEDRON
    6,  // WEDGE
    5,  // PYRAMID
    3,  // QUADRARIC_EDGE
    6,  // QUADRATIC_TRIANGLE
    8,  // QUADRATIC_QUAD
    10, // QUADRATIC_TETRA
    20  // QUADRARIC_HEXAHEDRON
};

// Element base class
// All FE should be derived from that class. The class provide interface for building stiffness,
// damping and inertia matrices (methods buildK(), buildC(), buildM()), for get element results
// (methods getScalar(...), getVector(...), getTensor(...)), for update element state after solution
// iteration (method update()).
class Element {
  public:
    Element();
    virtual ~Element();

    // get element number, as it is stored in FEStorage. Numbers start from 1.
    uint32 getElNum() const;
    // return number of nodes for the element
    uint16 getNNodes() const;
    // return number of dimensions (0D, 1D, 2D, 3D) occupied by element shape.
    uint16 getDim() const;
    // return element shape
    ElementShape getShape() const;
    // return element type
    ElementType getType() const;
    // return node number (as it stored in FEStorage) of the i-th node in the element
    uint32& getNodeNumber(uint16 num) const;
    // return the FEStorage to which element belongs
    FEStorage& getStorage() const;
    // return order of integration scheme used in the particular element for integration over volume
    uint16 getIntegrationOrder() const;
    // set the integration order for the element
    void setIntegrationOrder(uint16 _nint); // нельзя вызывать после выполнения функции pre() (начало решения)

    // heart of the element class
    // TODO: comment massively here
    virtual void pre() = 0;
    virtual void buildK() = 0;
    virtual void buildC();
    virtual void buildM();
    virtual void update() = 0;

    // The methods below are getters to receive solution information related to elements (like
    // stresses, strains, volume and so on). The first argument is a pointer on return value. Please
    // note, that result of the function times scale will be added to the pointer's value (*scalar
    // += result * scale, in particular case). query is a value from enum shows with particular
    // result should be returned. gp is a number of Gaussian point, if gp == GP_MEAN then the
    // result will be averaged over the element based on current integration scheme.
    // The methods return true if the query is relevant for the element and false if the element
    // can't return asked query code.
    virtual bool getScalar(double* scalar, scalarQuery query, uint16 gp = GP_MEAN, double scale = 1.0);
    virtual bool getVector(math::Vec<3>& vector, vectorQuery query, uint16 gp = GP_MEAN, double scale = 1.0);
    virtual bool getTensor(math::MatSym<3>& tensor, tensorQuery query, uint16 gp = GP_MEAN, double scale = 1.0);

    Element& operator=(const Element& from);

    // in-out operation:
    void print(std::ostream& out) const;

    // some general purpose assemble procedures. Particular element realization could have it own
    // assembly procedure.
    template <uint16 dimM> void assembleK(math::MatSym<dimM>& Ke, std::initializer_list<Dof::dofType> _nodeDofs);
    template <uint16 dimM> void assembleC(math::MatSym<dimM>& Ce, std::initializer_list<Dof::dofType> _nodeDofs);
    template <uint16 dimM> void assembleM(math::MatSym<dimM>& Me, std::initializer_list<Dof::dofType> _nodeDofs);

    template <uint16 dimM>
    void assembleK(math::MatSym<dimM>& Ke, math::Vec<dimM>& Fe, std::initializer_list<Dof::dofType> _nodeDofs);

    void assembleK(Eigen::Ref<Eigen::MatrixXd> Ke, std::initializer_list<Dof::dofType> _nodeDofs) const;

    friend class FEStorage;

  protected:
    ElementType type = ElementType::UNDEFINED;
    ElementShape shape = UNDEFINED;
    uint16 intOrder = 0; // number of int points overall
    uint32 elNum = 0;
    uint32* nodes = nullptr;
    FEStorage* storage = nullptr;
};

// Element geometry class
class ElementVERTEX : public Element {
  public:
    ElementVERTEX() {
        shape = VERTEX;
        nodes = new uint32[getNNodes()];
    }

    ElementVERTEX& operator=(const ElementVERTEX& from) = default;
};

class ElementTWIN_VERTEX : public Element {
  public:
    ElementTWIN_VERTEX() {
        shape = TWIN_VERTEX;
        nodes = new uint32[getNNodes()];
    }

    ElementTWIN_VERTEX& operator=(const ElementTWIN_VERTEX& from) = default;
};

class ElementLINE : public Element {
  public:
    ElementLINE() {
        shape = LINE;
        nodes = new uint32[getNNodes()];
    }

    ElementLINE& operator=(const ElementLINE& from) = default;
};

class ElementTRIANGLE : public Element {
  public:
    ElementTRIANGLE() {
        shape = TRIANGLE;
        nodes = new uint32[getNNodes()];
    }

    ElementTRIANGLE& operator=(const ElementTRIANGLE& from) = default;
};

class ElementQUAD : public Element {
  public:
    ElementQUAD() {
        shape = QUAD;
        nodes = new uint32[getNNodes()];
    }

    ElementQUAD& operator=(const ElementQUAD& from) = default;
};

class ElementTETRA : public Element {
  public:
    ElementTETRA() {
        shape = TETRA;
        nodes = new uint32[getNNodes()];
    }

    ElementTETRA& operator=(const ElementTETRA& from) = default;
};

class ElementHEXAHEDRON : public Element {
  public:
    ElementHEXAHEDRON() {
        shape = HEXAHEDRON;
        nodes = new uint32[getNNodes()];
    }

    ElementHEXAHEDRON& operator=(const ElementHEXAHEDRON& from) = default;
};

class ElementWEDGE : public Element {
  public:
    ElementWEDGE() {
        shape = WEDGE;
        nodes = new uint32[getNNodes()];
    }

    ElementWEDGE& operator=(const ElementWEDGE& from) = default;
};

} // namespace nla3d

// 'dirty' hack to avoid include loops (element-vs-festorage)
#include "FEStorage.h"

namespace nla3d {

template <uint16 dimM>
void Element::assembleK(math::MatSym<dimM>& Ke, const std::initializer_list<Dof::dofType> _nodeDofs) {
    assert(nodes != nullptr);
    const double* Ke_p = Ke.ptr();
    const std::vector<Dof::dofType> nodeDof(_nodeDofs);
    const auto dim = static_cast<uint16>(_nodeDofs.size());
    assert(getNNodes() * dim == dimM);

    for (uint16 i = 0; i < getNNodes(); i++) {
        for (uint16 di = 0; di < dim; di++) {
            for (uint16 j = i; j < getNNodes(); j++) {
                for (uint16 dj = 0; dj < dim; dj++) {
                    if (i == j && dj < di) {
                        continue;
                    }
                    storage->addValueK(nodes[i], nodeDof[di], nodes[j], nodeDof[dj], *Ke_p);
                    ++Ke_p;
                }
            }
        }
    }
}

template <uint16 dimM>
void Element::assembleC(math::MatSym<dimM>& Ce, const std::initializer_list<Dof::dofType> _nodeDofs) {
    assert(nodes != nullptr);
    const double* Ce_p = Ce.ptr();
    const std::vector<Dof::dofType> nodeDof(_nodeDofs);
    const auto dim = static_cast<uint16>(_nodeDofs.size());
    assert(getNNodes() * dim == dimM);

    for (uint16 i = 0; i < getNNodes(); i++) {
        for (uint16 di = 0; di < dim; di++) {
            for (uint16 j = i; j < getNNodes(); j++) {
                for (uint16 dj = 0; dj < dim; dj++) {
                    if ((i == j) && (dj < di)) {
                        continue;
                    }
                    storage->addValueC(nodes[i], nodeDof[di], nodes[j], nodeDof[dj], *Ce_p);
                    ++Ce_p;
                }
            }
        }
    }
}

template <uint16 dimM>
void Element::assembleM(math::MatSym<dimM>& Me, const std::initializer_list<Dof::dofType> _nodeDofs) {
    assert(nodes != nullptr);
    const double* Me_p = Me.ptr();
    const std::vector<Dof::dofType> nodeDof(_nodeDofs);
    const auto dim = static_cast<uint16>(_nodeDofs.size());
    assert(getNNodes() * dim == dimM);

    for (uint16 i = 0; i < getNNodes(); i++) {
        for (uint16 di = 0; di < dim; di++) {
            for (uint16 j = i; j < getNNodes(); j++) {
                for (uint16 dj = 0; dj < dim; dj++) {
                    if ((i == j) && (dj < di)) {
                        continue;
                    }
                    storage->addValueC(nodes[i], nodeDof[di], nodes[j], nodeDof[dj], *Me_p);
                    ++Me_p;
                }
            }
        }
    }
}

template <uint16 dimM>
void Element::assembleK(math::MatSym<dimM>& Ke, math::Vec<dimM>& Fe,
                        const std::initializer_list<Dof::dofType> _nodeDofs) {
    assert(nodes != nullptr);
    const double* Ke_p = Ke.ptr();
    const std::vector<Dof::dofType> nodeDof(_nodeDofs);
    const auto dim = static_cast<uint16>(_nodeDofs.size());
    assert(getNNodes() * dim == dimM);

    for (uint16 i = 0; i < getNNodes(); i++) {
        for (uint16 di = 0; di < dim; di++) {
            for (uint16 j = i; j < getNNodes(); j++) {
                for (uint16 dj = 0; dj < dim; dj++) {
                    if ((i == j) && (dj < di)) {
                        continue;
                    }
                    storage->addValueK(nodes[i], nodeDof[di], nodes[j], nodeDof[dj], *Ke_p);
                    ++Ke_p;
                }
            }
        }
    }

    const double* Fe_p = Fe.ptr();
    for (uint16 i = 0; i < getNNodes(); i++) {
        for (uint16 di = 0; di < dim; di++) {
            storage->addValueF(nodes[i], nodeDof[di], *Fe_p);
            ++Fe_p;
        }
    }
}

inline uint16 Element::getNNodes() const { return _shape_nnodes[shape]; }

inline uint16 Element::getDim() const { return _shape_dim[shape]; }

inline ElementShape Element::getShape() const { return shape; }

inline ElementType Element::getType() const { return type; }

// & is used here because this function is called such this:
// el->getNodeNumber(0) = 1234;
inline uint32& Element::getNodeNumber(const uint16 num) const {
    assert(num < getNNodes());
    assert(nodes);
    return nodes[num];
}

inline uint16 Element::getIntegrationOrder() const { return intOrder; }

inline void Element::setIntegrationOrder(const uint16 _nint) { intOrder = _nint; }

inline FEStorage& Element::getStorage() const { return *storage; }

inline uint32 Element::getElNum() const { return elNum; }

} // namespace nla3d
