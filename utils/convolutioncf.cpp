#include "convolutioncf.hpp"
#include <iostream>

namespace ngfem
{
  ConvolutionCoefficientFunction::ConvolutionCoefficientFunction (shared_ptr<CoefficientFunction> ac1,
                              shared_ptr<CoefficientFunction> ac2,
                              shared_ptr<ngcomp::MeshAccess> ama, int aorder)
  : CoefficientFunction(ac1->Dimension(), ac1->IsComplex()),
    c1(ac1), c2(ac2), ma(ama), order(aorder)
  {}

  ConvolutionCoefficientFunction::~ConvolutionCoefficientFunction ()
  {}

  void ConvolutionCoefficientFunction::PrintReport (ostream & ost) const
  {
    ost << "Convolve(";
    c1->PrintReport(ost);
    ost << ", ";
    c2->PrintReport(ost);
    ost << ")";
  }

  void ConvolutionCoefficientFunction::TraverseTree (const function<void(CoefficientFunction&)> & func)
  {
    c1->TraverseTree (func);
    c2->TraverseTree (func);
    func(*this);
  }

  double ConvolutionCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    int dim = c1->Dimension();
    auto point = ip.GetPoint();
    Vector<> sum(dim);
    sum = 0.0;
    auto glh = LocalHeap(1000000, "convolutioncf lh", true);

    ma->IterateElements
      (VOL, glh, [&] (ngcomp::Ngs_Element el, LocalHeap & lh)
        {
          auto & trafo = ma->GetTrafo (el, lh);
          Vector<> hsum(dim);
          hsum = 0.0;

          IntegrationRule ir(trafo.GetElementType(), order);
          BaseMappedIntegrationRule & mir = trafo(ir, lh);
          FlatMatrix<> vals1(ir.Size(), dim, lh);
          FlatMatrix<> vals2(ir.Size(), dim, lh);
          c1->Evaluate (mir, vals1);
          auto mirpts = mir.GetPoints();
          mirpts *= -1;
          for (int i = 0; i < mirpts.Height(); i++)
            mirpts.Row(i) += point;
          // at this point, mir probably has the wrong ElementTransformation
          // but it doesn't matter as long as c2 only uses the global points
          c2->Evaluate (mir, vals2);
          for (int i = 0; i < vals1.Height(); i++)
            hsum += mir[i].GetWeight() * vals1.Row(i) * vals2.Row(i);
          for(size_t i = 0; i<dim;i++)
          AsAtomic(sum(i)) += hsum(i);
        });

    return sum(0);
  }

  // double ConvolutionCoefficientFunction::EvaluateConst () const
  // {
  //   return c2->EvaluateConst();
  // }

  // void ConvolutionCoefficientFunction::Evaluate(const BaseMappedIntegrationPoint & ip,
  //                       FlatVector<> result) const
  // {
  //   IntegrationPoint outip;
  //   Vector<> res1(c1->Dimension());
  //   c1->Evaluate(ip, res1);
  //   int el = ma->FindElementOfPoint(res1, outip, false);
  //   if (el == -1)
  //   {
  //     result = 0;
  //     return;
  //   }
  //   LocalHeap lh(100000);
  //   BaseMappedIntegrationPoint mappedip(outip, ma->GetTrafo(el, lh));
  //   c2->Evaluate(mappedip, result);
  // }
}
