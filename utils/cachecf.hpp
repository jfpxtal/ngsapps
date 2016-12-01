#include <comp.hpp>
#include <python_ngstd.hpp>

namespace ngfem
{
  class CacheCoefficientFunction : public CoefficientFunction
  {
    shared_ptr<CoefficientFunction> c;
    shared_ptr<ngcomp::FESpace> fes;
    shared_ptr<SymbolicBilinearFormIntegrator> bfi;
    unique_ptr<IntegrationPointCoefficientFunction> ipcf;
  public:
    CacheCoefficientFunction (shared_ptr<CoefficientFunction> ac,
                              shared_ptr<ngcomp::FESpace> afes);
    virtual ~CacheCoefficientFunction () {}
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
    void SetBFI(shared_ptr<SymbolicBilinearFormIntegrator> abfi);
    void Refresh();
  };
}

namespace ngstd
{
  template <> struct PyWrapperTraits<ngfem::CacheCoefficientFunction> {
    typedef PyWrapperDerived<ngfem::CacheCoefficientFunction, ngfem::CoefficientFunction> type;
  };
  template <>
  struct PyWrapperTraits<ngfem::SymbolicBilinearFormIntegrator> {
    typedef PyWrapperClass<ngfem::SymbolicBilinearFormIntegrator> type;
  };
}
