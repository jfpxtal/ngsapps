#pragma once

#include <fem.hpp>

namespace ngfem
{

  class DummyElementTransformation : public ElementTransformation
  {
    int spacedim;
  public:
    DummyElementTransformation(int aspacedim) : ElementTransformation(ET_POINT, VOL, -1, -1), spacedim(aspacedim)   {}

    virtual void CalcJacobian (const IntegrationPoint & ip,
                              FlatMatrix<> dxdxi) const
    {
      throw Exception("DummyElementTransformation CalcJacobian");
    }

    /// calculate the mapped point
    virtual void CalcPoint (const IntegrationPoint & ip,
                            FlatVector<> point) const
    {
      throw Exception("DummyElementTransformation CalcPoint");
    }

    /// calculate point and Jacobi matrix
    virtual void CalcPointJacobian (const IntegrationPoint & ip,
                                    FlatVector<> point, FlatMatrix<> dxdxi) const
    {
      throw Exception("DummyElementTransformation CalcPointJacobian");
    }

    /// Calculate points and Jacobimatrices in all points of integrationrule
    virtual void CalcMultiPointJacobian (const IntegrationRule & ir,
                                        BaseMappedIntegrationRule & mir) const
    {
      throw Exception("DummyElementTransformation CalcMultiPointJacobian");
    }

    /// returns space dimension of physical elements
    virtual int SpaceDim () const
    {
      // throw Exception("DummyElementTransformation SpaceDim");
      return spacedim;
    }

    /// is it a mapping for boundary or codim 2 elements ?
    virtual VorB VB() const
    {
      // throw Exception("DummyElementTransformation VB");
      return VOL;
    }

    /// return a mapped integration point on localheap
    virtual BaseMappedIntegrationPoint & operator() (const IntegrationPoint & ip, Allocator & lh) const
    {
      throw Exception("DummyElementTransformation operator(IP)");
    }


    /// return a mapped integration rule on localheap
    virtual BaseMappedIntegrationRule & operator() (const IntegrationRule & ir, Allocator & lh) const
    {
      throw Exception("DummyElementTransformation operator(IR)");
    }

  };

}

const ngfem::BaseMappedIntegrationPoint &DummyMIPFromPoint(const ngbla::FlatVector<double> &points, ngstd::LocalHeap &lh);
