#pragma once

#include <fem.hpp>

namespace ngfem
{
  class AnnulusSpeedDx;
  class AnnulusSpeedDy;

  class AnnulusSpeedCoefficientFunction : public CoefficientFunction
  {
  protected:
    double Rinner, Router, phi0, vout, v0, smearR, smearphi;

  public:
    AnnulusSpeedCoefficientFunction(double aRinner, double aRouter, double aphi0, double avout, double av0, double asmearR, double asmearphi);
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    // virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;

    shared_ptr<AnnulusSpeedDx> Dx() const { return make_shared<AnnulusSpeedDx>(Rinner, Router, phi0, vout, v0, smearR, smearphi); }
    shared_ptr<AnnulusSpeedDy> Dy() const { return make_shared<AnnulusSpeedDy>(Rinner, Router, phi0, vout, v0, smearR, smearphi); }
  };

  class AnnulusSpeedDx : public AnnulusSpeedCoefficientFunction
  {
  public:
    using AnnulusSpeedCoefficientFunction::AnnulusSpeedCoefficientFunction;
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    // virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
  };

  class AnnulusSpeedDy : public AnnulusSpeedCoefficientFunction
  {
  public:
    using AnnulusSpeedCoefficientFunction::AnnulusSpeedCoefficientFunction;
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    // virtual void Evaluate (const SIMD_BaseMappedIntegrationRule & ir, BareSliceMatrix<SIMD<double>> values) const;
  };
}
