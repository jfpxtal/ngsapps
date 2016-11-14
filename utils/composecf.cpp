#include "composecf.hpp"

namespace ngfem
{
  ComposeCoefficientFunction::ComposeCoefficientFunction (shared_ptr<CoefficientFunction> ac1,
                              shared_ptr<CoefficientFunction> ac2,
                              shared_ptr<ngcomp::MeshAccess> ama)
  : CoefficientFunction(ac2->Dimension(), ac2->IsComplex()),
      c1(ac1), c2(ac2), ma(ama)
  {}

  ComposeCoefficientFunction::~ComposeCoefficientFunction ()
  {}

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
    int el = ma->FindElementOfPoint(res1, outip, false);
    if (el == -1) return 0;
    LocalHeap lh(100000);
    BaseMappedIntegrationPoint mappedip(outip, ma->GetTrafo(el, lh));
    return c2->Evaluate(mappedip);
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
    int el = ma->FindElementOfPoint(res1, outip, false);
    if (el == -1)
    {
      result = 0;
      return;
    }
    LocalHeap lh(100000);
    BaseMappedIntegrationPoint mappedip(outip, ma->GetTrafo(el, lh));
    c2->Evaluate(mappedip, result);
  }
}
