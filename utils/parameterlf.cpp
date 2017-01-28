#include "parameterlf.hpp"

#include "utils.hpp"

namespace ngfem
{
  ParameterLinearFormCF::ParameterLinearFormCF (shared_ptr<CoefficientFunction> aintegrand,
                                                                  shared_ptr<ngcomp::GridFunction> agf,
                                                                  int aorder)
  : CoefficientFunction(aintegrand->Dimension(), aintegrand->IsComplex()),
    integrand(aintegrand), gf(agf), order(aorder),
    LUT(agf->GetFESpace()->GetMeshAccess()->GetNE()), SIMD_LUT(agf->GetFESpace()->GetMeshAccess()->GetNE())
  {
    if (integrand->Dimension() != 1)
      throw Exception ("ParameterLinearFormCF needs scalar-valued CoefficientFunction");
    integrand->TraverseTree
      ([&] (CoefficientFunction & nodecf)
       {
         auto proxy = dynamic_cast<ProxyFunction*> (&nodecf);
         if (proxy && !proxies.Contains(proxy))
           proxies.Append (proxy);
       });
    if (proxies.Size() != 1)
      throw Exception ("ParameterLinearFormCF: the integrand has to contain exactly one proxy: the test function of gf's FESpace");
  }

  ParameterLinearFormCF::~ParameterLinearFormCF ()
  {}

  void ParameterLinearFormCF::PrintReport (ostream & ost) const
  {
    ost << "ParameterLF(";
    integrand->PrintReport(ost);
    ost << ", ";
    gf->PrintReport(ost);
    ost << ")";
  }

  void ParameterLinearFormCF::TraverseTree (const function<void(CoefficientFunction&)> & func)
  {
    integrand->TraverseTree (func);
    func(*this);
  }

  double ParameterLinearFormCF::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    throw Exception("ParameterLinearFormCF::Evaluate IP");
  }

  void ParameterLinearFormCF :: Evaluate (const BaseMappedIntegrationRule & ir,
                                                FlatMatrix<double> values) const
  {
    // TODO: test test test

    auto &lutElEntry = LUT[ir.GetTransformation().GetElementNr()];
    auto &irSizeMap = lutElEntry.first;
    shared_lock<shared_timed_mutex> readLock(lutElEntry.second);
    auto it = irSizeMap.find(ir.Size());
    if (it == irSizeMap.end())
    //if (1)
    {
      readLock.unlock();
      unique_lock<shared_timed_mutex> writeLock(lutElEntry.second);
      // check again?
      auto lh = LocalHeap(100000, "parameterlf lh", true);
      //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl << endl << endl;
      //cout << "ir " << ir.IR() << endl << endl;
      auto fes = gf->GetFESpace();
      // //cout << "cache miss " << ir.GetTransformation().GetElementNr() << " " << ir.Size() << endl;
      it = irSizeMap.emplace(ir.Size(), Matrix<double>(ir.Size(), fes->GetNDof())).first;
      it->second = 0;
      auto points = ir.GetPoints();
      //cout << "points " << points << endl << endl;
      ParameterLFUserData ud;
      ud.paramCoords.AssignMemory(fes->GetSpacialDimension(), lh);
      // FlatVector<double> gfvec(fes->GetNDof(), lh);
      // fes->IterateElements doesn't work if Evaluate was called inside a TaskManager task
      // because TaskManager gets stuck on nested tasks
      for (const auto &el : fes->Elements(VOL, lh))
      {
        //cout << "element " << el << endl << endl;
        auto & trafo = el.GetTrafo();
        const_cast<ElementTransformation&>(trafo).userdata = &ud;
        IntegrationRule myIR(el.GetType(), order);
        // IntegrationRule myIR(el.GetType(), 2*el.GetFE().Order());
        //cout << "fel order" << el.GetFE().Order() << endl << endl;
        //cout << "myIR " << myIR << endl << endl;
        auto & myMIR = trafo(myIR, lh);
        int elvec_size = el.GetFE().GetNDof()*fes->GetDimension();
        //cout << "elvec_size " << elvec_size << endl << endl;
        FlatVector<double> elvec(elvec_size, lh);
        FlatVector<double> elvec1(elvec_size, lh);
        auto dnums = el.GetDofs();
        //cout << "dnums " << dnums << endl << endl;
        for (auto j : Range(ir.Size()))
        {
          elvec = 0;
          for (auto l : Range(fes->GetSpacialDimension()))
            ud.paramCoords(l) = points(j, l);
          //cout << "ud " << ud.paramCoords << endl << endl;

          for (auto proxy : proxies)
          {
            FlatMatrix<double> proxyvalues(myIR.Size(), proxy->Dimension(), lh);
            for (int k = 0; k < proxy->Dimension(); k++)
            {
              //cout << "k " << k << endl;
              ud.testfunction = proxy;
              ud.test_comp = k;

              FlatMatrix<double> intVals(myMIR.Size(), 1, lh);
              integrand->Evaluate(myMIR, intVals);
              //cout << "intvals " << intVals << endl << endl;
              for (size_t i = 0; i < myMIR.Size(); i++) {
                //cout << "i " << i << " weight " << myMIR[i].GetWeight() << " " << intVals(i, 0) << endl << endl;
                proxyvalues(i, k) = myMIR[i].GetWeight() * intVals(i, 0);
              }
            }
            //cout << "proxvals " << proxyvalues << endl << endl;

            proxy->Evaluator()->ApplyTrans(el.GetFE(), myMIR, proxyvalues, elvec1, lh);
            //cout << "elv1 " << elvec1 << endl << endl;
            elvec += elvec1;
          }

          //cout << "elvec " << elvec << endl << endl;
          // fes->TransformVec(el, elvec, ngcomp::TRANSFORM_RHS);
          for (int k = 0; k < dnums.Size(); k++)
            if (dnums[k] != -1)
              it->second(j, dnums[k]) += elvec(k);
        }
      }
      //cout << "mat " << it->second << endl << endl;
      //cout << "gf vec " << gf->GetVector().FVDouble() << endl << endl;
      values = it->second * gf->GetVector().FVDouble();
      //cout << "res " << values << endl << endl;
    } else {
      values = it->second * gf->GetVector().FVDouble();
    }
  }

  void ParameterLinearFormCF :: Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const
  {
    // TODO: test test test
    //cout << "                SIMD !!!!!!!!!!!!!!!" << endl << endl;

    auto &lutElEntry = SIMD_LUT[ir.GetTransformation().GetElementNr()];
    auto &irSizeMap = lutElEntry.first;
    shared_lock<shared_timed_mutex> readLock(lutElEntry.second);
    auto it = irSizeMap.find(ir.Size());
    if (it == irSizeMap.end())
    //if (1)
    {
      readLock.unlock();
      unique_lock<shared_timed_mutex> writeLock(lutElEntry.second);
      // check again?
      auto lh = LocalHeap(100000, "parameterlf lh", true);
      //cout << ir.IR() << endl << endl;
      //cout << ir << endl << endl;
      auto fes = gf->GetFESpace();
      // cout << "cache miss " << ir.GetTransformation().GetElementNr() << " " << ir.Size() << endl;
      it = irSizeMap.emplace(ir.Size(), Matrix<SIMD<double>>(ir.Size(), fes->GetNDof())).first;
      it->second = SIMD<double>(0);
      auto points = ir.GetPoints();
      ParameterLFUserData ud;
      ud.paramCoords.AssignMemory(fes->GetSpacialDimension(), lh);
      // FlatVector<double> gfvec(fes->GetNDof(), lh);
      // fes->IterateElements doesn't work if Evaluate was called inside a TaskManager task
      // because TaskManager gets stuck on nested tasks
      for (const auto &el : fes->Elements(VOL, lh))
      {
        //cout << "element " << i << endl << endl;
        auto & trafo = el.GetTrafo();
        const_cast<ElementTransformation&>(trafo).userdata = &ud;
        SIMD_IntegrationRule myIR(el.GetType(), order);
        //cout << "myIR " << myIR << endl << endl;
        auto & myMIR = trafo(myIR, lh);
        int elvec_size = el.GetFE().GetNDof()*fes->GetDimension();
        FlatVector<double> elvec(elvec_size, lh);
        auto dnums = el.GetDofs();
        for (auto j : Range(ir.Size()))
        {
          for (auto m : Range(SIMD<IntegrationPoint>::Size())) // not good
          {
            elvec = 0;
            for (auto l : Range(fes->GetSpacialDimension()))
              ud.paramCoords(l) = points.Get(j, l)[m];

            for (auto proxy : proxies)
            {
              FlatMatrix<SIMD<double>> proxyvalues(proxy->Dimension(), myIR.Size(), lh);
              for (int k = 0; k < proxy->Dimension(); k++)
              {
                ud.testfunction = proxy;
                ud.test_comp = k;

                integrand->Evaluate(myMIR, proxyvalues.Rows(k,k+1));
                for (size_t i = 0; i < myMIR.Size(); i++)
                  proxyvalues(k,i) *= myMIR[i].GetWeight();
              }

              proxy->Evaluator()->AddTrans(el.GetFE(), myMIR, proxyvalues, elvec);
            }

            fes->TransformVec(el, elvec, ngcomp::TRANSFORM_RHS);
            for (int k = 0; k < dnums.Size(); k++)
              if (dnums[k] != -1)
                it->second(j, dnums[k]) += SIMD<double>([k, m, &elvec](int i) { return i==m ? elvec(k) : 0; }); // yikes
          }
        }
      }
      values.AddSize(Dimension(), ir.Size()) = Trans(it->second * gf->GetVector().FVDouble());

    } else {
      values.AddSize(Dimension(), ir.Size()) = Trans(it->second * gf->GetVector().FVDouble());
    }
  }

}
