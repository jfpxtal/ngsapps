#pragma once

#include <bla.hpp>
#include <comp.hpp>
#include <python_ngstd.hpp>

#include <shared_mutex>

namespace ngfem
{

  class ConvolutionCoefficientFunction : public CoefficientFunction
  {
    shared_ptr<CoefficientFunction> cf;
    shared_ptr<CoefficientFunction> kernel;
    shared_ptr<ngcomp::MeshAccess> ma;
    int order;

    // IntegrationRule sizes for the given order, summed over all elements
    int totalConvIRSize;
    int SIMD_totalConvIRSize;

    // lookup table for kernel values
    // ASSUMPTION: IntegrationRules given as input of Evaluate() during the lifetime
    // of this ConvolutionCF can be uniquely identified by their Size() and the
    // corresponding element
    mutable vector<pair<map<int, typename ngbla::Matrix<>>, shared_timed_mutex>> kernelLUT;
    // TODO: do we really need two different LUTs?
    mutable vector<pair<map<int, typename ngbla::Matrix<SIMD<double>>>, shared_timed_mutex>> SIMD_kernelLUT;
    vector<typename ngbla::Matrix<SIMD<double>>> cfLUT;
  public:
    ConvolutionCoefficientFunction (shared_ptr<CoefficientFunction> acf,
                                    shared_ptr<CoefficientFunction> akernel,
                                    shared_ptr<ngcomp::MeshAccess> ama, int aorder);
    virtual ~ConvolutionCoefficientFunction ();
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void Evaluate (const BaseMappedIntegrationRule & ir,
                           FlatMatrix<double> values) const;
    virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
    // virtual double EvaluateConst () const;
    // virtual void Evaluate (const BaseMappedIntegrationPoint & ip, FlatVector<> result) const;
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
    void CacheCF();
    void ClearCFCache() { cfLUT.clear(); }
  };

  class LambdaCoefficientFunction : public CoefficientFunction
  {
    std::function<double(double, double)> func;

  public:
    LambdaCoefficientFunction(std::function<double(double, double)> f) : CoefficientFunction(1), func(std::move(f)) {}

    virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const
    {
      auto point = ip.GetPoint();
      return func(point[0], point[1]);
    }
  };

  class GaussCoefficientFunction : public CoefficientFunction
  {
  protected:
    double scal, var;
  public:
    GaussCoefficientFunction(double ascal, double avar) : CoefficientFunction(1), scal(ascal), var(avar) {}
  };

  class GaussKernelDx : public GaussCoefficientFunction
  {
  public:
    using GaussCoefficientFunction::GaussCoefficientFunction;

    virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const
      {
        auto point = ip.GetPoint();
        auto &x = point[0];
        auto &y = point[1];
        return -2*x*scal*var*exp(-var * (x*x + y*y));
      }
  };

  class GaussKernelDy : public GaussCoefficientFunction
  {
  public:
    using GaussCoefficientFunction::GaussCoefficientFunction;

    virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const
      {
        auto point = ip.GetPoint();
        auto &x = point[0];
        auto &y = point[1];
        return -2*y*scal*var*exp(-var * (x*x + y*y));
      }
  };

  class GaussKernel : public GaussCoefficientFunction
  {
  public:
    using GaussCoefficientFunction::GaussCoefficientFunction;

    virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const
    {
      auto point = ip.GetPoint();
      return scal * exp(-var * (point[0]*point[0] + point[1]*point[1]));
    }

    std::shared_ptr<CoefficientFunction> Dx() const { return std::make_shared<GaussKernelDx>(scal, var); }
    std::shared_ptr<CoefficientFunction> Dy() const { return std::make_shared<GaussKernelDy>(scal, var); }

    // std::shared_ptr<LambdaCoefficientFunction> Dx() const
    // {
    //   auto lscal = scal;
    //   auto lvar = var;
    //   return std::make_shared<LambdaCoefficientFunction>([lscal, lvar](auto x, auto y) {
    //       return -2*x*lscal*lvar*exp(-lvar * (x*x + y*y));
    //     });
    // }

    // std::shared_ptr<LambdaCoefficientFunction> Dy() const
    //   {
    //     auto lscal = scal;
    //     auto lvar = var;
    //     return std::make_shared<LambdaCoefficientFunction>([lscal, lvar](auto x, auto y) {
    //           return -2*y*lscal*lvar*exp(-lvar * (x*x + y*y));
    //         });
    //   }
  };

}
