#include "zlogzcf.hpp"
#include <limits>

namespace ngfem
{
  ZLogZCoefficientFunction::ZLogZCoefficientFunction (shared_ptr<CoefficientFunction> acf)
    : CoefficientFunction(1), cf(acf)
  { }

  ///
  double ZLogZCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    double z = cf->Evaluate(ip);
    // if (z < 0) {
    //   return -1*std::numeric_limits<double>::infinity();
    // } else if (z == 0) {
    if (z <= 0) {
      return 0;
    } else {
      return z*log(z);
    }
  }
}
