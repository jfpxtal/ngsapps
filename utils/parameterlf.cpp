#include "parameterlf.hpp"

#include "utils.hpp"

namespace ngfem
{

  CompactlySupportedKernel::CompactlySupportedKernel(double aradius, double ascale)
    : CoefficientFunction(1, false), radius(aradius), scale(ascale)
  {}

  double CompactlySupportedKernel::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    const auto &p = ip.GetPoint();
    const auto &paramCoords = static_cast<ParameterLFUserData*>(ip.GetTransformation().userdata)->paramCoords;
    auto x = p[0] - paramCoords[0];
    auto y = p[1] - paramCoords[1];
    double d =  1 - sqrt(x*x+y*y)/radius;
    if (d > 0) return scale*d;
    else return 0;
  }

  void CompactlySupportedKernel::Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const
  {
    const auto &ps = ir.GetPoints();
    const auto &paramCoords = static_cast<ParameterLFUserData*>(ir.GetTransformation().userdata)->paramCoords;
    for (int k = 0; k < ir.Size(); k++)
    {
      auto x = ps.Get(k, 0) - paramCoords[0];
      auto y = ps.Get(k, 1) - paramCoords[1];
      auto d =  FMA(-1/radius, sqrt(x*x+y*y), 1);
      values(0, k) = scale*ngstd::IfPos(d, d, 0);
    }
  }


  ParameterLinearFormCF::ParameterLinearFormCF (shared_ptr<CoefficientFunction> aintegrand,
                                                shared_ptr<ngcomp::GridFunction> agf,
                                                int aorder, int arepeat, vector<double> apatchSize)
  : CoefficientFunction(aintegrand->Dimension(), aintegrand->IsComplex()),
    integrand(aintegrand), gf(agf), order(aorder), repeat(arepeat), patchSize(apatchSize),
    fes(gf->GetFESpace()), numPatches(1),
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

    for (int i = 0; i < fes->GetSpacialDimension(); i++) numPatches *= 2*repeat + 1;
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
      auto lh = LocalHeap(100000, "parameterlf lh");
      //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl << endl << endl;
      //cout << "ir " << ir.IR() << endl << endl;
      // //cout << "cache miss " << ir.GetTransformation().GetElementNr() << " " << ir.Size() << endl;
      it = irSizeMap.emplace(ir.Size(), Matrix<double>(ir.Size(), fes->GetNDof())).first;
      it->second = 0;
      auto points = ir.GetPoints();
      //cout << "points " << points << endl << endl;
      ParameterLFUserData ud;
      ud.paramCoords.AssignMemory(fes->GetSpacialDimension(), lh);
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
        const auto &periodicMIR = MakePeriodicMIR<BaseMappedIntegrationRule>(myMIR, lh);

        auto & fel = el.GetFE();
        int elvec_size = fel.GetNDof()*fes->GetDimension();
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
            proxyvalues = 0;
            for (int k = 0; k < proxy->Dimension(); k++)
            {
              //cout << "k " << k << endl;
              ud.testfunction = proxy;
              ud.test_comp = k;

              FlatMatrix<double> intVals(periodicMIR.Size(), 1, lh);

              integrand->Evaluate(periodicMIR, intVals);
              //cout << "intvals " << intVals << endl << endl;
              for (int patch = 0; patch < numPatches; patch++)
              {
                for (size_t i = 0; i < myMIR.Size(); i++)
                {
                  proxyvalues(i, k) += myMIR[i].GetWeight() * intVals(patch*myMIR.Size() + i, 0);
                  //cout << "i " << i << " weight " << myMIR[i].GetWeight() << " " << intVals(i, 0) << endl << endl;
                }
              }
            }
            //cout << "proxvals " << proxyvalues << endl << endl;

            proxy->Evaluator()->ApplyTrans(fel, myMIR, proxyvalues, elvec1, lh);
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
      // cout << "mat " << it->second << endl << endl;
      // cout << "gf vec " << gf->GetVector().FVDouble() << endl << endl;
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
      auto lh = LocalHeap(100000, "parameterlf lh");
      //cout << ir.IR() << endl << endl;
      //cout << ir << endl << endl;
      // cout << "cache miss " << ir.GetTransformation().GetElementNr() << " " << ir.Size() << endl;
      it = irSizeMap.emplace(ir.Size(), Matrix<SIMD<double>>(ir.Size(), fes->GetNDof())).first;
      it->second = SIMD<double>(0);
      auto points = ir.GetPoints();
      ParameterLFUserData ud;
      ud.paramCoords.AssignMemory(fes->GetSpacialDimension(), lh);
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
        const auto &periodicMIR = MakePeriodicMIR<SIMD_BaseMappedIntegrationRule>(myMIR, lh);

        auto & fel = el.GetFE();
        int elvec_size = fel.GetNDof()*fes->GetDimension();
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
              proxyvalues = 0;
              for (int k = 0; k < proxy->Dimension(); k++)
              {
                ud.testfunction = proxy;
                ud.test_comp = k;

                FlatMatrix<SIMD<double>> intVals(1, periodicMIR.Size(), lh);

                // integrand->Evaluate(periodicMIR, proxyvalues.Rows(k,k+1));
                integrand->Evaluate(periodicMIR, intVals);
                for (int patch = 0; patch < numPatches; patch++)
                {
                  for (size_t i = 0; i < myMIR.Size(); i++)
                  {
                    proxyvalues(k, i) += myMIR[i].GetWeight() * intVals(0, patch*myMIR.Size() + i);
                    //cout << "i " << i << " weight " << myMIR[i].GetWeight() << " " << intVals(i, 0) << endl << endl;
                  }
                }
              }

              proxy->Evaluator()->AddTrans(fel, myMIR, proxyvalues, elvec);
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
