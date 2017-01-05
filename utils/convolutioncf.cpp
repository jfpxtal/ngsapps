#include "convolutioncf.hpp"

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
    auto lh = LocalHeap(100000, "convolutioncf lh", true);
    // ma->IterateElements doesn't work if Evaluate was called inside a TaskManager task
    // because TaskManager gets stuck on nested tasks
    for (auto i : Range(ma->GetNE()))
    {
      HeapReset hr(lh);
      ElementId ei(VOL, i);
      auto & trafo = ma->GetTrafo (ei, lh);
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
        sum(i) += hsum(i);
    }

    return sum(0);
  }

  void ConvolutionCoefficientFunction :: Evaluate (const BaseMappedIntegrationRule & ir,
                                                FlatMatrix<double> values) const
  {
    auto points = ir.GetPoints();
    auto lh = LocalHeap(100000, "convolutioncf lh", true);
    values = 0;
    // ma->IterateElements doesn't work if Evaluate was called inside a TaskManager task
    // because TaskManager gets stuck on nested tasks
    for (auto i : Range(ma->GetNE()))
    {
      HeapReset hr(lh);
      ElementId ei(VOL, i);
      auto & trafo = ma->GetTrafo (ei, lh);

      IntegrationRule convIR(trafo.GetElementType(), order);
      BaseMappedIntegrationRule & convMIR = trafo(convIR, lh);
      FlatMatrix<> vals1(convIR.Size(), 1, lh);
      c1->Evaluate (convMIR, vals1);
      auto mirpts = convMIR.GetPoints();

      for (auto j : Range(ir.Size()))
      {
        for (auto k : Range(convMIR.Size()))
        {
          FlatVector<double> newpt = points.Row(j) - mirpts.Row(k) | lh;
          // trafo is probably the wrong ElementTransformation for the new point
          // but it doesn't matter as long as c2 only uses the actual point, not the trafo
          BaseMappedIntegrationPoint *mip;
          switch (trafo.SpaceDim())
          {
          case 1: {
            auto dmip = new (lh) DimMappedIntegrationPoint<1>(IntegrationPoint(), trafo);
            dmip->Point() = newpt;
            mip = dmip;
            break; }
          case 2: {
            auto dmip = new (lh) DimMappedIntegrationPoint<2>(IntegrationPoint(), trafo);
            dmip->Point() = newpt;
            mip = dmip;
            break; }
          case 3: {
            auto dmip = new (lh) DimMappedIntegrationPoint<3>(IntegrationPoint(), trafo);
            dmip->Point() = newpt;
            mip = dmip;
            break; }
          }
          values(j, 0) += convMIR[k].GetWeight() * vals1(k) * c2->Evaluate(*mip);
        }
      }

    }
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
