// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once

#include "math/Mat.h"
#include "math/SparseMatrix.h"
#include "sys.h"

namespace nla3d {

namespace math {

class SparseSymMatrix;

// EquationSolver - abstract class for solving a system of linear equations.
// This class primarly used in FESolver class.

class EquationSolver {
  public:
    virtual ~EquationSolver() = default;
    virtual void solveEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) = 0;
    virtual void factorizeEquations(math::SparseSymMatrix* matrix) = 0;
    virtual void substituteEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) = 0;
    void setSymmetric(bool symmetric = true);
    void setPositive(bool positive = true);

  protected:
    uint32 nEq = 0;

    // number of rhs
    int nrhs = 1;
    bool isSymmetric = true;
    bool isPositive = true;
};

class GaussDenseEquationSolver : public EquationSolver {
  public:
    ~GaussDenseEquationSolver() override = default;
    void solveEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) override;
    void factorizeEquations(math::SparseSymMatrix* matrix) override;
    void substituteEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) override;
    static bool _solve(double* X, double* A, double* B, uint32 n);

  protected:
    dMat matA = dMat(1, 1);
};

struct ConjugateGradientEquationSolver : EquationSolver {
    ~ConjugateGradientEquationSolver() override = default;
    void solveEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) override;
    void factorizeEquations(math::SparseSymMatrix* matrix) override;
    void substituteEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) override;

    explicit ConjugateGradientEquationSolver(const double tolerance = 2.220446049250313e-16, const uint32 max_iters = 0)
        : tolerance(tolerance), maxIters(max_iters) {}

    double tolerance;
    /// Specifies maximum iterations of CG; value `0` means auto
    uint32 maxIters;
};

class BiCGSTAB_EquationSolver : public EquationSolver {
  public:
    ~BiCGSTAB_EquationSolver() override = default;
    void solveEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) override;
    void factorizeEquations(math::SparseSymMatrix* matrix) override;
    void substituteEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns) override;
};

#ifdef NLA3D_USE_MKL
class PARDISO_equationSolver : public EquationSolver {
  public:
    virtual ~PARDISO_equationSolver();
    virtual void solveEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns);
    virtual void factorizeEquations(math::SparseSymMatrix* matrix);
    virtual void substituteEquations(math::SparseSymMatrix* matrix, double* rhs, double* unknowns);

  protected:
    void initializePARDISO(math::SparseSymMatrix* matrix);
    void releasePARDISO();

    // Internal solver memory pointer pt
    void* pt[64];
    // Paramaters for PARDISO solver (see MKL manual for clarifications)
    int iparm[64];

    // maximum number of numerical factorizations
    int maxfct = 1;

    // which factorization to use
    int mnum = 1;
    // don't print statistical information in file
    int msglvl = 0;

    bool firstRun = true;
    // real symmetric undifinite defined matrix
    int mtype = -2;
};
#endif // NLA3D_USE_MKL

extern EquationSolver* defaultEquationSolver;

} // namespace math

} // namespace nla3d
