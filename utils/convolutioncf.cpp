#include "convolutioncf.hpp"

#include "utils.hpp"

namespace ngfem
{
  ConvolutionCoefficientFunction::ConvolutionCoefficientFunction (shared_ptr<CoefficientFunction> acf,
                                                                  shared_ptr<CoefficientFunction> akernel,
                                                                  shared_ptr<ngcomp::MeshAccess> ama, int aorder)
  : CoefficientFunction(acf->Dimension(), acf->IsComplex()),
    cf(acf), kernel(akernel), ma(ama), order(aorder),
    kernelLUT(ma->GetNE()), totalConvIRSize(0), SIMD_kernelLUT(ma->GetNE()), SIMD_totalConvIRSize(0)
  {
    // TODO: switch to FESpace as argument, use FESpace->Elements
    for (auto &&el : ma->Elements())
    {
      totalConvIRSize += IntegrationRule(el.GetType(), order).Size();
      SIMD_totalConvIRSize += SIMD_IntegrationRule(el.GetType(), order).Size()*SIMD<IntegrationPoint>::Size();
    }
  }

  ConvolutionCoefficientFunction::~ConvolutionCoefficientFunction ()
  {}

  void ConvolutionCoefficientFunction::PrintReport (ostream & ost) const
  {
    ost << "Convolve(";
    cf->PrintReport(ost);
    ost << ", ";
    kernel->PrintReport(ost);
    ost << ")";
  }

  void ConvolutionCoefficientFunction::TraverseTree (const function<void(CoefficientFunction&)> & func)
  {
    cf->TraverseTree (func);
    kernel->TraverseTree (func);
    func(*this);
  }

  double ConvolutionCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    // TODO: kernelLUT
    int dim = cf->Dimension();
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
      cf->Evaluate (mir, vals1);
      auto mirpts = mir.GetPoints();
      mirpts *= -1;
      for (int i = 0; i < mirpts.Height(); i++)
        mirpts.Row(i) += point;
      // at this point, mir probably has the wrong ElementTransformation
      // but it doesn't matter as long as kernel only uses the global points
      // TODO: ip.SetNr(-666)
      kernel->Evaluate (mir, vals2);
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
    // TODO: test test test
    auto lh = LocalHeap(100000, "convolutioncf lh", true);
    values = 0;

    auto &lutElEntry = kernelLUT[ir.GetTransformation().GetElementNr()];
    auto &irSizeMap = lutElEntry.first;
    shared_lock<shared_timed_mutex> readLock(lutElEntry.second);
    auto it = irSizeMap.find(ir.Size());
    if (it == irSizeMap.end())
    {
      readLock.unlock();
      unique_lock<shared_timed_mutex> writeLock(lutElEntry.second);
      //cout << ir.IR() << endl << endl;

      // cout << "cache miss " << ir.GetTransformation().GetElementNr() << " " << ir.Size() << endl;
      auto &mat = irSizeMap.emplace(ir.Size(), Matrix<>(ir.Size(), totalConvIRSize)).first->second;
      auto points = ir.GetPoints();
      int col = 0;
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

        cf->Evaluate (convMIR, vals1);

        auto mirpts = convMIR.GetPoints();

        for (auto j : Range(ir.Size()))
        {
          for (auto k : Range(convIR.Size()))
          {
            FlatVector<double> newpt = points.Row(j) - mirpts.Row(k) | lh;
            // dummy MIP doesn't have the correct ElementTransformation
            // but it doesn't matter as long as kernel only uses the actual point, not the trafo
            // should be ok for most convolution kernels
            auto val2 = convMIR[k].GetWeight() * kernel->Evaluate(DummyMIPFromPoint(newpt, lh));
            mat(j, col+k) = val2;
            values(j, 0) += vals1(k) * val2;
          }
        }
        col += convIR.Size();
      }

    } else {
      int col = 0;
      for (auto i : Range(ma->GetNE()))
      {
        HeapReset hr(lh);
        ElementId ei(VOL, i);
        auto & trafo = ma->GetTrafo (ei, lh);
        IntegrationRule convIR(trafo.GetElementType(), order);
        auto & convMIR = trafo(convIR, lh);
        FlatMatrix<> vals1(convIR.Size(), 1, lh);

        cf->Evaluate(convMIR, vals1);
        values += it->second.Cols(col, col+convIR.Size()) * vals1;
        col += convIR.Size();
      }
    }
  }

  void PrintBare(const ABareMatrix<double> &mat, int h, int w)
  {
    cout << "Bare " << h << " " << w << endl;
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++)
        cout << " " << mat.Get(i, j);
      cout << endl;
    }
    cout << endl << endl;
  }

  void ConvolutionCoefficientFunction::CacheCF()
  {
    cfLUT.resize(ma->GetNE());
    LocalHeap glh(100000, "convolution cachecf lh");
    ma->IterateElements
      (VOL, glh, [&] (ngcomp::Ngs_Element el, LocalHeap & lh)
       {
         auto & trafo = ma->GetTrafo (el, lh);
         SIMD_IntegrationRule convIR(trafo.GetElementType(), order);
         auto & convMIR = trafo(convIR, lh);
         cfLUT[el.Nr()].SetSize(1, convIR.Size());

         cf->Evaluate(convMIR, cfLUT[el.Nr()]);
       });
  }


  void ConvolutionCoefficientFunction :: Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const
  {
    // TODO: test test test
    //cout << "                SIMD !!!!!!!!!!!!!!!" << endl << endl;
    auto lh = LocalHeap(100000, "convolutioncf lh", true);
    values.AddSize(Dimension(), ir.Size()) = 0;

    auto &lutElEntry = SIMD_kernelLUT[ir.GetTransformation().GetElementNr()];
    auto &irSizeMap = lutElEntry.first;
    shared_lock<shared_timed_mutex> readLock(lutElEntry.second);
    auto it = irSizeMap.find(ir.Size());
    if (it == irSizeMap.end())
    //if (1)
    {
      readLock.unlock();
      unique_lock<shared_timed_mutex> writeLock(lutElEntry.second);
      //cout << ir.IR() << endl << endl;
      //cout << ir << endl << endl;

      // cout << "cache miss " << ir.GetTransformation().GetElementNr() << " " << ir.Size() << endl;
      auto &mat = irSizeMap.emplace(ir.Size(), Matrix<SIMD<double>>(SIMD_totalConvIRSize, ir.Size())).first->second;
      auto points = ir.GetPoints();
      int row = 0;
      // ma->IterateElements doesn't work if Evaluate was called inside a TaskManager task
      // because TaskManager gets stuck on nested tasks
      for (auto i : Range(ma->GetNE()))
      {
        //cout << "element " << i << endl << endl;
        HeapReset hr(lh);
        ElementId ei(VOL, i);
        auto & trafo = ma->GetTrafo (ei, lh);

        SIMD_IntegrationRule convIR(trafo.GetElementType(), order);
        //cout << "convIR " << convIR << endl << endl;
        auto & convMIR = trafo(convIR, lh);
        //cout << "convMIR " << convMIR << endl << endl;
        const FlatMatrix<SIMD<double>> *vals1;
        if (! cfLUT.empty()) vals1 = &cfLUT[i];
        else {
          vals1 = new (lh) FlatMatrix<SIMD<double>>(1, convIR.Size(), lh);

          cf->Evaluate(convMIR, *vals1);
        }
        //cout << "vals1 " << vals1 << endl << endl;

        auto mirpts = convMIR.GetPoints();
        //cout << "points" << endl;
        //PrintBare(points, ir.Size(), 2);
        //cout << "mirpts" << endl;
        //PrintBare(mirpts, convMIR.Size(), 2);

        size_t newSize = ir.Size()*convIR.Size()*SIMD<IntegrationPoint>::Size();
        FlatArray<SIMD<IntegrationPoint>> newIPs(newSize, lh);
        // dummy MIP doesn't have the correct ElementTransformation
        // but it doesn't matter as long as kernel only uses the actual points, not the trafo
        // should be ok for most convolution kernels
        SIMD_BaseMappedIntegrationRule *newBMIR;
        switch (ir.DimSpace())
        {
        case 1:
          newBMIR = new (lh) SIMD_MappedIntegrationRule<0, 1>(SIMD_IntegrationRule(newSize, &newIPs[0]), DummyElementTransformation(1), -1, lh);
          break;
        case 2:
          newBMIR = new (lh) SIMD_MappedIntegrationRule<0, 2>(SIMD_IntegrationRule(newSize, &newIPs[0]), DummyElementTransformation(2), -1, lh);
          break;
        case 3:
          newBMIR = new (lh) SIMD_MappedIntegrationRule<0, 3>(SIMD_IntegrationRule(newSize, &newIPs[0]), DummyElementTransformation(3), -1, lh);
          break;
        }

        for (auto j : Range(convIR.Size()))
        {
          for (auto m : Range(SIMD<IntegrationPoint>::Size()))
          {
            for (auto k : Range(ir.Size()))
            {
              for (auto l : Range(ir.DimSpace()))
                newBMIR->GetPoints().Get(j*SIMD<IntegrationPoint>::Size()*ir.Size()+m*ir.Size()+k, l) = points.Get(k, l) - mirpts.Get(j, l)[m];
            }
          }
        }

        //cout << "newBMIR " << *newBMIR << endl << endl;
        kernel->Evaluate(*newBMIR, mat.Rows(row, row+SIMD<IntegrationPoint>::Size()*convIR.Size()));
        //cout << "vals 2 " << endl << mat.Rows(row, SIMD<IntegrationPoint>::Size()*row+convIR.Size()) << endl << endl;
        //cout << "weights" << endl;
        for (auto j : Range(convIR.Size()))
        {
          for (auto m : Range(SIMD<IntegrationPoint>::Size()))
          {
            mat.Row(row+j*SIMD<IntegrationPoint>::Size()+m) *= convMIR[j].GetWeight()[m];
            for (auto k : Range(ir.Size()))
              values(0, k) += (*vals1)(j)[m] * mat(row+j*SIMD<IntegrationPoint>::Size()+m, k);
            // values(0, k) += vals1(j)[m] * mat(row+j*SIMD<IntegrationPoint>::Size()+m, k);
          }
          // cout << k << ": " << convMIR[k].GetMeasure() << " " << convMIR[k].IP().Weight() << " " << "w " <<  convMIR[k].GetWeight() << endl;
        }
        //cout << endl << endl;
        //cout << mat.Rows(row, row+SIMD<IntegrationPoint>::Size()*convIR.Size()) << endl << endl;

        // values.AddSize(Dimension(), ir.Size()) += vals1 * mat.Rows(row, row+SIMD<IntegrationPoint>::Size()*convIR.Size());
        //cout << values.AddSize(Dimension(), ir.Size()) << endl << endl;

        row += SIMD<IntegrationPoint>::Size()*convIR.Size();
      }

    } else {
      int row = 0;
      for (auto i : Range(ma->GetNE()))
      {
        HeapReset hr(lh);
        ElementId ei(VOL, i);
        auto & trafo = ma->GetTrafo (ei, lh);
        SIMD_IntegrationRule convIR(trafo.GetElementType(), order);
        const FlatMatrix<SIMD<double>> *vals1;
        if (! cfLUT.empty()) vals1 = &cfLUT[i];
        else {
          auto & convMIR = trafo(convIR, lh);
          vals1 = new (lh) FlatMatrix<SIMD<double>>(1, convIR.Size(), lh);

          cf->Evaluate(convMIR, *vals1);
        }

        for (auto j : Range(convIR.Size()))
        {
          for (auto m : Range(SIMD<IntegrationPoint>::Size()))
          {
            for (auto k : Range(ir.Size()))
              values(0, k) += (*vals1)(j)[m] * it->second(row+j*SIMD<IntegrationPoint>::Size()+m, k);
          }
        }
        row += SIMD<IntegrationPoint>::Size()*convIR.Size();
      }
    }
  }

  // double ConvolutionCoefficientFunction::EvaluateConst () const
  // {
  //   return kernel->EvaluateConst();
  // }

  // void ConvolutionCoefficientFunction::Evaluate(const BaseMappedIntegrationPoint & ip,
  //                       FlatVector<> result) const
  // {
  //   IntegrationPoint outip;
  //   Vector<> res1(cf->Dimension());
  //   cf->Evaluate(ip, res1);
  //   int el = ma->FindElementOfPoint(res1, outip, false);
  //   if (el == -1)
  //   {
  //     result = 0;
  //     return;
  //   }
  //   LocalHeap lh(100000);
  //   BaseMappedIntegrationPoint mappedip(outip, ma->GetTrafo(el, lh));
  //   kernel->Evaluate(mappedip, result);
  // }
}
