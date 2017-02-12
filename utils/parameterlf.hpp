#pragma once

#include <bla.hpp>
#include <comp.hpp>
#include <python_ngstd.hpp>

#include <shared_mutex>

namespace ngfem
{
  class ParameterLFUserData : public ProxyUserData
  {
  public:
    FlatVector<double> paramCoords;
  };

  class ParameterLFProxy : public CoefficientFunction
  {
    int dir;

  public:
    ParameterLFProxy(int adir) : CoefficientFunction(1, false), dir(adir) {}
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const
    {
      return static_cast<ParameterLFUserData*>(ip.GetTransformation().userdata)->paramCoords[dir];
    }
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const
    {
      values.AddSize(1, ir.Size()) = static_cast<ParameterLFUserData*>(ir.GetTransformation().userdata)->paramCoords[dir];
    }
  };

  // too slow in python:
  class CompactlySupportedKernel : public CoefficientFunction
  {
    double radius, scale;

  public:
    CompactlySupportedKernel(double aradius, double ascale);
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
  };


  class ParameterLinearFormCF : public CoefficientFunction
  {
    shared_ptr<CoefficientFunction> integrand;
    shared_ptr<ngcomp::GridFunction> gf;
    int order, repeat;
    vector<double> patchSize;

    Array<ProxyFunction*> proxies;
    shared_ptr<ngcomp::FESpace> fes;
    int numPatches;

    // lookup tables
    // ASSUMPTION: IntegrationRules given as input of Evaluate() during the lifetime
    // of this coefficient function can be uniquely identified by their Size() and the
    // corresponding element
    mutable vector<pair<map<int, typename ngbla::Matrix<double>>, shared_timed_mutex>> LUT;
    mutable vector<pair<map<int, typename ngbla::Matrix<SIMD<double>>>, shared_timed_mutex>> SIMD_LUT;


    template <class TRAITS>
    typename TRAITS::BMIR &T_MakePeriodicMIR(const typename TRAITS::BMIR &bmir, LocalHeap &lh) const;
    template <class BMIR>
    BMIR &MakePeriodicMIR(BMIR &bmir, LocalHeap &lh) const;


  public:
    ParameterLinearFormCF (shared_ptr<CoefficientFunction> aintegrand,
                           shared_ptr<ngcomp::GridFunction> agf,
                           int aorder, int arepeat=0, vector<double> apatchSize={});
    virtual ~ParameterLinearFormCF ();
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void Evaluate (const BaseMappedIntegrationRule & ir,
                           FlatMatrix<double> values) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
  };

  template <class BMIR, int DIM> struct IntegrationTraits;

  template <int DIM>
  struct IntegrationTraits<BaseMappedIntegrationRule, DIM>
  {
    static const int dim = DIM;
    typedef double SCAL;
    typedef IntegrationPoint IP;
    typedef IntegrationRule IR;
    typedef BaseMappedIntegrationRule BMIR;
    typedef MappedIntegrationPoint<DIM, DIM> MIP;
    typedef MappedIntegrationRule<DIM, DIM> MIR;
  };

  template <int DIM>
  struct IntegrationTraits<SIMD_BaseMappedIntegrationRule, DIM>
  {
    static const int dim = DIM;
    typedef SIMD<double> SCAL;
    typedef SIMD<IntegrationPoint> IP;
    typedef SIMD_IntegrationRule IR;
    typedef SIMD_BaseMappedIntegrationRule BMIR;
    typedef SIMD<MappedIntegrationPoint<DIM, DIM>> MIP;
    typedef SIMD_MappedIntegrationRule<DIM, DIM> MIR;
  };


  template <class TRAITS>
  typename TRAITS::BMIR &ParameterLinearFormCF::T_MakePeriodicMIR(
    const typename TRAITS::BMIR &bmir, LocalHeap &lh) const
  {
    // typename TRAITS::IR periodicIR;
    FlatArray<typename TRAITS::IP> periodicIPs(numPatches*bmir.Size(), lh);
    for (int patch = 0; patch < numPatches; patch++)
    {
      for (int i = 0; i < bmir.Size(); i++)
        periodicIPs[patch*bmir.Size() + i] = bmir[i].IP();
    }

    auto &res = *(new (lh) typename TRAITS::MIR(typename TRAITS::IR(numPatches*bmir.Size(), &periodicIPs[0]), bmir.GetTransformation(), -1, lh));

    Vec<TRAITS::dim, int> curPatchIdx(-repeat);
    Vec<TRAITS::dim, typename TRAITS::SCAL> newPoint;
    for (int patch = 0;; patch++)
    {
      for (int i = 0; i < bmir.Size(); i++)
      {
        newPoint = bmir[i].GetPoint();
        for (int j = 0; j < TRAITS::dim; j++)
          newPoint(j) += curPatchIdx(j)*patchSize[j];
        res[patch*bmir.Size() + i] = static_cast<const typename TRAITS::MIR&>(bmir)[i];
        res[patch*bmir.Size() + i].Point() = newPoint;
      }
      int i = 0;
      for (; i < TRAITS::dim; i++)
      {
        if (curPatchIdx(i) != repeat) break;
      }
      if (i == TRAITS::dim) break;
      curPatchIdx(i)++;
      for (; i > 0; i--) curPatchIdx(i-1) = -repeat;
    }

    return res;
  }

  template <class BMIR>
  BMIR &ParameterLinearFormCF::MakePeriodicMIR(
    BMIR &bmir, LocalHeap &lh) const
  {
    if (repeat == 0) return bmir;

    switch (bmir.GetTransformation().SpaceDim())
    {
    case 1:
      return T_MakePeriodicMIR<IntegrationTraits<BMIR, 1>>(bmir, lh);
    case 2:
      return T_MakePeriodicMIR<IntegrationTraits<BMIR, 2>>(bmir, lh);
    case 3:
      return T_MakePeriodicMIR<IntegrationTraits<BMIR, 3>>(bmir, lh);
    default:
      throw Exception("ParameterLinearFormCF::MakePeriodicMIR invalid SpaceDim");
    }
  }


}
