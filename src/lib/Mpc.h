// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "Dof.h"
#include "sys.h"
#include <list>

namespace nla3d {

class FEStorage;

class MpcTerm {
  public:
    MpcTerm() : coef(1.0) {}

    MpcTerm(const int32 n, const Dof::dofType dof, const double coef = 1.0) : node(n), node_dof(dof), coef(coef) {}

    int32 node = 0;
    Dof::dofType node_dof = Dof::UNDEFINED;
    double coef = 0.0;
};

class Mpc {
  public:
    void print(std::ostream& out);
    std::list<MpcTerm> eq;
    double b;     // rhs of MPC eq.
    uint32 eqNum; // number of equation in global assembly
};

class MpcCollection {
  public:
    virtual ~MpcCollection() = default;
    virtual void update() = 0;
    virtual void pre() = 0;
    void printEquations(std::ostream& out) const;
    void registerMpcsInStorage() const;
    std::vector<Mpc*> collection;
    FEStorage* storage;
};

class RigidBodyMpc : public MpcCollection {
  public:
    RigidBodyMpc();
    ~RigidBodyMpc() override;
    uint32 masterNode;

    void pre() override;
    void update() override;
    std::vector<uint32> slaveNodes;
    std::vector<Dof::dofType> dofs;
};

class fixBC {
  public:
    fixBC() = default;
    fixBC(const int32 n, const Dof::dofType dof, const double val = 0.0) : node(n), node_dof(dof), value(val) {}
    int32 node = 0;
    Dof::dofType node_dof = Dof::UNDEFINED;
    double value = 0.0;
};

class loadBC {
  public:
    loadBC() = default;
    loadBC(const int32 n, const Dof::dofType dof, const double val = 0.0) : node(n), node_dof(dof), value(val) {}
    int32 node = 0;
    Dof::dofType node_dof = Dof::UNDEFINED;
    double value = 0.0;
};

} // namespace nla3d
