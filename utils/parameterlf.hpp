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
  class PeriodicCompactlySupportedKernel : public CoefficientFunction
  {
    double dx, dy, radius, scale;

  public:
    PeriodicCompactlySupportedKernel(double adx, double ady, double aradius, double ascale);
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
  };


  class ParameterLinearFormCF : public CoefficientFunction
  {
    shared_ptr<CoefficientFunction> integrand;
    shared_ptr<ngcomp::GridFunction> gf;
    int order;

    Array<ProxyFunction*> proxies;

    // lookup tables
    // ASSUMPTION: IntegrationRules given as input of Evaluate() during the lifetime
    // of this coefficient function can be uniquely identified by their Size() and the
    // corresponding element
    mutable vector<pair<map<int, typename ngbla::Matrix<double>>, shared_timed_mutex>> LUT;
    mutable vector<pair<map<int, typename ngbla::Matrix<SIMD<double>>>, shared_timed_mutex>> SIMD_LUT;
  public:
    ParameterLinearFormCF (shared_ptr<CoefficientFunction> aintegrand,
                                    shared_ptr<ngcomp::GridFunction> agf,
                                    int aorder);
    virtual ~ParameterLinearFormCF ();
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void Evaluate (const BaseMappedIntegrationRule & ir,
                           FlatMatrix<double> values) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
  };

}
