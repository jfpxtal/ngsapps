#include <comp.hpp>

void project(shared_ptr<ngcomp::GridFunction> gf, shared_ptr<ngcomp::GridFunction> res);
void limit(shared_ptr<ngcomp::GridFunction>, shared_ptr<ngcomp::FESpace>, double, double, double, bool nonneg);
void limitold(shared_ptr<ngcomp::GridFunction>, shared_ptr<ngcomp::FESpace>, double, double, double, bool nonneg);
