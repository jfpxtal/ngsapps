#include "utils.hpp"

using namespace ngfem;

const BaseMappedIntegrationPoint &DummyMIPFromPoint(const FlatVector<double> &point, LocalHeap &lh)
{
  BaseMappedIntegrationPoint *mip;
  IntegrationPoint dummyIP;
  dummyIP.SetNr(-666); // used by cachecf to detect dummy mip
  auto dummyET = new (lh) ngfem::DummyElementTransformation(point.Size());
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
