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
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
    void CacheCF();
    void ClearCFCache() { cfLUT.clear(); }
  };

}
