#include "eikonal.hpp"

using namespace ngcomp;

void solveEikonal1D(shared_ptr<CoefficientFunction> rhs, shared_ptr<GridFunction> res)
{
  auto fes = res->GetFESpace();
  auto ir = IntegrationRule(ET_SEGM, 2*fes->GetOrder());
  LocalHeap lh(100000, "eikonal lh");
  vector<double> lvals, rvals;
  bool first = true;
  double lastp, lastv, rightbnd, sum = 0;
  int ne = 0;
  for (const auto &el : fes->Elements())
  {
    HeapReset hr(lh);
    ne++;
    const auto &trafo = el.GetTrafo();
    auto &mir = trafo(ir, lh);
    FlatVector<> rhsvals(ir.Size(), lh);
    rhs->Evaluate(mir, rhsvals.AsMatrix(ir.Size(), 1));
    if (first)
      lastp = trafo(IntegrationPoint(1), lh).GetPoint()[0];
    rightbnd = trafo(IntegrationPoint(0), lh).GetPoint()[0];
    for (int i = mir.Size()-1; i >= 0; i--)
    {
      if (first)
        first = false;
      else
        rvals.emplace_back((mir[i].GetPoint()[0]-lastp) * lastv);
      lastv = rhsvals[i];
      sum += (mir[i].GetPoint()[0]-lastp) * lastv;
      lvals.emplace_back(sum);
      lastp = mir[i].GetPoint()[0];
    }
  }
  rvals.emplace_back((rightbnd-lastp) * lastv);
  partial_sum(rvals.rbegin(), rvals.rend(), rvals.rbegin());
  Array<double> resvals(lvals.size());
  for (int j : Range(ne))
  {
    for (int i : Range(ir))
      resvals[j*ir.Size()+i] = min(lvals[(j+1)*ir.Size()-1-i], rvals[(j+1)*ir.Size()-1-i]);
  }

  auto ipcf = make_shared<IntegrationPointCoefficientFunction>(ne, ir.Size(), resvals);
  SetValues(ipcf, *res, VOL, nullptr, lh);
}
