#include <comp.hpp>

void solveEikonal1D(shared_ptr<ngfem::CoefficientFunction> rhs, shared_ptr<ngcomp::GridFunction> res);

class EikonalSolver2D
{
  typedef vector<pair<int, double>> DistByPnt;
  shared_ptr<ngcomp::MeshAccess> ma;
  vector<DistByPnt> distByPntByRef;
  vector<array<double, 3>> anglesByEl;
  shared_ptr<ngcomp::FESpace> nodal_fes;
  shared_ptr<ngcomp::GridFunction> nodal_gf;

public:
  EikonalSolver2D(shared_ptr<ngcomp::FESpace>, const vector<Vec<2>> &);
  void solve(shared_ptr<ngfem::CoefficientFunction>);
  shared_ptr<ngcomp::GridFunction> getSolutionGF() const { return nodal_gf; }
};
