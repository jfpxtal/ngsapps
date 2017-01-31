#include <comp.hpp>
#include <python_ngstd.hpp>

namespace ngfem
{
  class ZLogZCoefficientFunction : public CoefficientFunction
  {
    shared_ptr<CoefficientFunction> cf;
    shared_ptr<ngcomp::MeshAccess> ma;
  public:
    ZLogZCoefficientFunction (shared_ptr<CoefficientFunction> acf);
    virtual ~ZLogZCoefficientFunction () {}
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    // void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
  };
}

namespace ngstd
{
  template <> struct PyWrapperTraits<ngfem::ZLogZCoefficientFunction> {
    typedef PyWrapperDerived<ngfem::ZLogZCoefficientFunction, ngfem::CoefficientFunction> type;
  };
}
