#pragma once

#include <fem.hpp>
#include <random>

namespace ngfem
{
  /// The coefficient evaluates random between two bounds (default: 0 and 1) pointwise
  class NGS_DLL_HEADER RandomCoefficientFunction : public CoefficientFunction
  {
    double lower_bound = 0;
    double upper_bound = 1;
    std::random_device r;
    mutable std::uniform_real_distribution<double> unif;
    mutable std::default_random_engine re;
  public:
    ///
    RandomCoefficientFunction (double lower, double upper);
    ///
    virtual ~RandomCoefficientFunction () {}
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual double EvaluateConst () const;
    virtual void Evaluate (const BaseMappedIntegrationRule & ir, FlatMatrix<double> values) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, AFlatMatrix<double> values) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, 
                           FlatArray<AFlatMatrix<double>*> input,
                           AFlatMatrix<double> values) const;
    virtual void PrintReport (ostream & ost) const;
  };
}
