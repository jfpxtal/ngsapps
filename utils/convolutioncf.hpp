#include <comp.hpp>
#include <python_ngstd.hpp>

namespace ngfem
{
  // class MyCoordCoefficientFunction : public CoefficientFunction
  // {
  // public:
  //   MyCoordCoefficientFunction (int dim) : CoefficientFunction(dim, false) {}
  //   virtual ~MyCoordCoefficientFunction () {}
  //   ///
  //   virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
  //   virtual void Evaluate (const BaseMappedIntegrationPoint & ip, FlatVector<> result) const;
  //   virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
  //   virtual void PrintReport (ostream & ost) const;
  // };

  class ConvolutionCoefficientFunction : public CoefficientFunction
  {
    shared_ptr<CoefficientFunction> c1;
    shared_ptr<CoefficientFunction> c2;
    shared_ptr<ngcomp::MeshAccess> ma;
    int order;
  public:
    ConvolutionCoefficientFunction (shared_ptr<CoefficientFunction> ac1,
                                    shared_ptr<CoefficientFunction> ac2,
                                    shared_ptr<ngcomp::MeshAccess> ama, int aorder);
    virtual ~ConvolutionCoefficientFunction ();
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void Evaluate (const BaseMappedIntegrationRule & ir,
                           FlatMatrix<double> values) const;
    // virtual double EvaluateConst () const;
    // virtual void Evaluate (const BaseMappedIntegrationPoint & ip, FlatVector<> result) const;
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
  };

  class GaussKernel : public CoefficientFunction
  {
  private:
    double scal, var;
  public:
    GaussKernel(double ascal, double avar) : CoefficientFunction(1), scal(ascal), var(avar) {}

    virtual double Evaluate(const BaseMappedIntegrationPoint &ip) const
    {
      auto point = ip.GetPoint();
      return scal * exp(-var * (point[0]*point[0] + point[1]*point[1]));
    }
  };

}
