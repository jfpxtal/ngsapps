#include "annulusspeedcf.hpp"

#include <math.h>

namespace ngfem
{
  AnnulusSpeedCoefficientFunction::AnnulusSpeedCoefficientFunction(double aRinner, double aRouter,
                                                                   double aphi0, double avout,
                                                                   double av0, double asmearR, double asmearphi)
    : CoefficientFunction(1, false), Rinner(aRinner), Router(aRouter), phi0(aphi0), vout(avout), v0(av0),
      smearR(asmearR), smearphi(asmearphi)
  {}

  double AnnulusSpeedCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    auto p = ip.GetPoint();
    auto radius = sqrt(p[0]*p[0] + p[1]*p[1]);

    auto angle = atan2(p[1], p[0])*180/M_PI;
    double res;
    if (-90-phi0/2 < angle && angle < -90+phi0/2)
      res = v0/2 * (1 - 2*(angle+90)/phi0);
    else if (90-phi0/2 < angle && angle < 90+phi0/2)
      res = v0/2 * (1 - 2*(angle-90)/phi0);
    else if (-90+phi0/2 <= angle && angle < -90+phi0/2+smearphi)
      res = v0 * (angle+90-phi0/2)/smearphi;
    else if (90+phi0/2 <= angle && angle < 90+phi0/2+smearphi)
      res = v0 * (angle-90-phi0/2)/smearphi;
    else res = v0;

    if (Rinner <= radius && radius <= Router)
      return res;
    else if (Rinner-smearR < radius && radius < Rinner)
      return res + (Rinner-radius)/smearR*(vout-res);
    else if (Router+smearR > radius && radius > Router)
      return res + (radius-Router)/smearR*(vout-res);
    else return vout;
  }
// virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;

  double AnnulusSpeedDx::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    auto p = ip.GetPoint();
    auto radius = sqrt(p[0]*p[0] + p[1]*p[1]);
    // TODO: not differentiable at boundaries
    if (radius < Rinner || radius > Router) return 0;

    auto angle = atan2(p[1], p[0])*180/M_PI;
    if (-90-phi0/2 < angle && angle < -90+phi0/2)
      return v0 * p[1]/(p[0]*p[0]+p[1]*p[1])/phi0 * 180/M_PI;
    else if (90-phi0/2 < angle && angle < 90+phi0/2)
      return v0 * p[1]/(p[0]*p[0]+p[1]*p[1])/phi0 * 180/M_PI;
    else return 0;
  }

  double AnnulusSpeedDy::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    auto p = ip.GetPoint();
    auto radius = sqrt(p[0]*p[0] + p[1]*p[1]);
    // TODO: not differentiable at boundaries
    if (radius < Rinner || radius > Router) return 0;

    auto angle = atan2(p[1], p[0])*180/M_PI;
    if (-90-phi0/2 < angle && angle < -90+phi0/2)
      return -v0 * p[0]/(p[0]*p[0]+p[1]*p[1])/phi0 * 180/M_PI;
    else if (90-phi0/2 < angle && angle < 90+phi0/2)
      return -v0 * p[0]/(p[0]*p[0]+p[1]*p[1])/phi0 * 180/M_PI;
    else return 0;
  }
}
