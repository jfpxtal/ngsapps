#pragma once

#include <fem.hpp>

const ngfem::BaseMappedIntegrationPoint &DummyMIPFromPoint(const ngbla::FlatVector<double> &points, ngstd::LocalHeap &lh);
