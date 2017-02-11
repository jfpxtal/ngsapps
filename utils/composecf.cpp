#include "composecf.hpp"

#include "utils.hpp"

namespace ngfem
{
  ComposeCoefficientFunction::ComposeCoefficientFunction (shared_ptr<CoefficientFunction> ac1,
                                                          shared_ptr<CoefficientFunction> ac2,
                                                          shared_ptr<ngcomp::MeshAccess> ama)
  : CoefficientFunction(ac2->Dimension(), ac2->IsComplex()),
    c1(ac1), c2(ac2), ma(ama)
  {
    if (ma)
    {
      // force mesh to build search tree to speed up later calls to FindElementOfPoint
      // can't call FindElementOfPoint with build_searchtree=true inside Evaluate
      // because Evaluate might get called from functions like VisualSceneSolution::DrawScene
      // which lock the mesh mutex and cause BuidSearchTree to get stuck
      IntegrationPoint dummy;
      ma->FindElementOfPoint(Vector<>({0, 0, 0}), dummy, true);
    }
  }

  void ComposeCoefficientFunction::PrintReport (ostream & ost) const
  {
    c1->PrintReport(ost);
    ost << "(";
    c2->PrintReport(ost);
    ost << ")";
  }

  void ComposeCoefficientFunction::TraverseTree (const function<void(CoefficientFunction&)> & func)
  {
    c1->TraverseTree (func);
    c2->TraverseTree (func);
    func(*this);
  }

  double ComposeCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    IntegrationPoint outip;
    Vector<> res1(c1->Dimension());
    c1->Evaluate(ip, res1);
    if (ma)
    {
      LocalHeap lh(100000);
      int el = ma->FindElementOfPoint(res1, outip, false);
      if (el == -1) return 0;
      ElementTransformation & eltrans = ma->GetTrafo(el, lh);
      BaseMappedIntegrationPoint & mip = eltrans(outip, lh);
      return c2->Evaluate(mip);
    } else {
      LocalHeap lh(1000);
      return c2->Evaluate(DummyMIPFromPoint(res1, lh));
    }
  }

  double ComposeCoefficientFunction::EvaluateConst () const
  {
    return c2->EvaluateConst();
  }

  void ComposeCoefficientFunction::Evaluate(const BaseMappedIntegrationPoint & ip,
                                            FlatVector<> result) const
  {
    IntegrationPoint outip;
    Vector<> res1(c1->Dimension());
    c1->Evaluate(ip, res1);
    if (ma)
    {
      LocalHeap lh(100000);
      int el = ma->FindElementOfPoint(res1, outip, false);
      if (el == -1)
      {
        result = 0;
        return;
      }
      ElementTransformation & eltrans = ma->GetTrafo(el, lh);
      BaseMappedIntegrationPoint & mip = eltrans(outip, lh);
      c2->Evaluate(mip, result);
    } else {
      // TODO: heap allocation too slow
      LocalHeap lh(1000);
      c2->Evaluate(DummyMIPFromPoint(res1, lh), result);
    }
  }
}
