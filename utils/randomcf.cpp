#include "randomcf.hpp"

namespace ngfem
{
  RandomCoefficientFunction::RandomCoefficientFunction (double lower, double upper)
    : CoefficientFunction(1), lower_bound(lower), upper_bound(upper),
      unif(lower_bound,upper_bound), re(r())
  { ; }

  ///
  double RandomCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    return unif(re);
  }

  double RandomCoefficientFunction::EvaluateConst () const
  {
    return unif(re);
  }
    
  void RandomCoefficientFunction::Evaluate (const BaseMappedIntegrationRule & ir, FlatMatrix<double> values) const
  {
    for (int i = 0; i < values.Height(); ++i)
      for (int j = 0; j < values.Width(); ++j)
        values(i,j) = unif(re);
  }

  void RandomCoefficientFunction::Evaluate (const SIMD_BaseMappedIntegrationRule & ir, AFlatMatrix<double> values) const
  { 
    for (int i = 0; i < values.Height(); ++i)
      for (int j = 0; j < values.Width(); ++j)
        values(i,j) = unif(re);
  }

  void RandomCoefficientFunction::Evaluate (const SIMD_BaseMappedIntegrationRule & ir, 
                                            FlatArray<AFlatMatrix<double>*> input,
                                            AFlatMatrix<double> values) const
  { 
    for (int i = 0; i < values.Height(); ++i)
      for (int j = 0; j < values.Width(); ++j)
        values(i,j) = unif(re);
  }

  void RandomCoefficientFunction::PrintReport (ostream & ost) const
  {
    ost << "CoefficientFunction is random between" << lower_bound << " and " << upper_bound << endl;
  }
}
