#include "utils.hpp"

using namespace ngfem;

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
    return spacedim;
  }

  /// is it a mapping for boundary or codim 2 elements ?
  virtual VorB VB() const
  {
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

const BaseMappedIntegrationPoint &DummyMIPFromPoint(const FlatVector<double> &point, LocalHeap &lh)
{
  BaseMappedIntegrationPoint *mip;
  IntegrationPoint dummyIP;
  dummyIP.SetNr(-666); // used by cachecf to detect dummy mip
  auto dummyET = new (lh) DummyElementTransformation(point.Size());
  switch (point.Size())
  {
  case 1: {
    auto dmip = new (lh) DimMappedIntegrationPoint<1>(dummyIP, *dummyET);
    dmip->Point() = point;
    mip = dmip;
    break; }
  case 2: {
    auto dmip = new (lh) DimMappedIntegrationPoint<2>(dummyIP, *dummyET);
    dmip->Point() = point;
    mip = dmip;
    break; }
  case 3: {
    auto dmip = new (lh) DimMappedIntegrationPoint<3>(dummyIP, *dummyET);
    dmip->Point() = point;
    mip = dmip;
    break; }
  }
  return *mip;
}
