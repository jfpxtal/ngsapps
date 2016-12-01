#include "cachecf.hpp"

namespace ngfem
{

  CacheCoefficientFunction::CacheCoefficientFunction (shared_ptr<CoefficientFunction> ac,
                                                      shared_ptr<ngcomp::FESpace> afes)
  : CoefficientFunction(ac->Dimension(), ac->IsComplex()),
    c(ac), fes(afes)
  {}

  void CacheCoefficientFunction::PrintReport (ostream & ost) const
  {
    ost << "Cache(";
    c->PrintReport(ost);
    ost << ")";
  }

  void CacheCoefficientFunction::TraverseTree (const function<void(CoefficientFunction&)> & func)
  {
    c->TraverseTree (func);
    func(*this);
  }

  void CacheCoefficientFunction::SetBFI (shared_ptr<SymbolicBilinearFormIntegrator> abfi)
  {
    bfi = abfi;
    int maxips = 0;
    int ne = 0;
    for (const auto & el : fes->Elements())
    {
      ne++;
      IntegrationRule ir = bfi->GetIntegrationRule(el.GetFE());
      if (ir.Size() > maxips)
        maxips = ir.Size();
    }
    ipcf = make_unique<IntegrationPointCoefficientFunction>(ne, maxips);
  }

  void CacheCoefficientFunction::Refresh ()
  {
    auto glh = LocalHeap(1000000, "cachecf lh", true);

    IterateElements
      (*fes, VOL, glh, [&] (ngcomp::FESpace::Element el, LocalHeap & lh)
        {
          IntegrationRule ir = bfi->GetIntegrationRule(el.GetFE());
          const ElementTransformation & trafo = el.GetTrafo();
          for (int i = 0; i < ir.Size(); i++)
            (*ipcf)(el.Nr(), i) = c->Evaluate(trafo(ir[i], lh));;
        });
  }

  double CacheCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    return ipcf->Evaluate(ip);
  }
}
