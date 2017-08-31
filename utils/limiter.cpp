#include "limiter.hpp"
#include <iostream>

using namespace std;
using namespace ngcomp;

double minmod(double v1, double v2)
{
  if (v1 > 0 && v2 > 0) return min(v1, v2);
  else if (v1 < 0 && v2 < 0) return max(v1, v2);
  else return 0;
}

double minmod_TVB(double v1, double v2, double M, double h)
{
  if (abs(v1) < M*h*h) return v1;
  else return minmod(v1, v2);
}

double posPart(double x)
{
  return (x > 0)*x;
}

double negPart(double x)
{
  return -(x < 0)*x;
}

void project(const FESpace::Element &el, const BaseMappedIntegrationRule &mir, const FlatMatrix<> &shape, const FlatVector<> &coeffs, shared_ptr<GridFunction> res, LocalHeap &lh)
{
  const auto &fel = el.GetFE();
  FlatMatrix<> comb_shape = Trans(shape)*coeffs | lh;
  for (int i : Range(mir))
    comb_shape(i) = mir[i].GetWeight()*comb_shape(i);

  FlatVector<> vec(fel.GetNDof(), lh);
  res->GetFESpace()->GetEvaluator(VOL)->ApplyTrans(fel, mir, comb_shape, vec, lh);

  FlatVector<> diag_mass(fel.GetNDof(), lh);
  switch (mir.GetTransformation().SpaceDim())
  {
  case 1:
    static_cast<const DGFiniteElement<1>&> (fel).GetDiagMassMatrix (diag_mass);
  case 2:
    static_cast<const DGFiniteElement<2>&> (fel).GetDiagMassMatrix (diag_mass);
  case 3:
    static_cast<const DGFiniteElement<3>&> (fel).GetDiagMassMatrix (diag_mass);
  }

  if (mir.GetTransformation().IsCurvedElement())
    throw Exception("limiter project curved el");

  IntegrationRule ir(fel.ElementType(), 0);
  BaseMappedIntegrationRule & mmir = mir.GetTransformation()(ir, lh);
  diag_mass *= mmir[0].GetMeasure();

  for (int i = 0; i < fel.GetNDof(); i++)
    vec[i] /= diag_mass[i];

  res->SetElementVector(el.GetDofs(), vec);
}

namespace std
{
  template <>
  struct iterator_traits<FlatVector<>::Iterator>
  {
    typedef forward_iterator_tag iterator_category;
  };
}

// Assuming L2HighOrderFESpace
void limit(shared_ptr<GridFunction> u, shared_ptr<FESpace> p1fes, double theta, double M, double h, bool nonneg)
{
  LocalHeap glh(100000, "limiter lh");
  const auto fes = u->GetFESpace();
  const auto ma = fes->GetMeshAccess();
  Flags flags;
  flags.SetFlag("novisual");
  auto p1gf = CreateGridFunction(p1fes, "gfu", flags);
  p1gf->Update();
  auto mass = make_shared<MassIntegrator<2>>(make_shared<ConstantCoefficientFunction>(1.0));
  // auto mass = GetIntegrators().CreateBFI('Mass', ma)
  CalcFluxProject(*u, *p1gf, mass, false, -1, glh);

  IterateElements(*fes, VOL, glh, [&] (FESpace::Element el, LocalHeap &lh)
    {
      auto &trafo = el.GetTrafo();
      const auto &fel = el.GetFE();
      auto &xT_mip = trafo(IntegrationPoint(1.0/3, 1.0/3), lh);
      auto xT = xT_mip.GetPoint();

      const auto &facets = el.Facets();
      const auto &et_edges = ElementTopology::GetEdges(el.GetType());
      const auto &et_verts = ElementTopology::GetVertices(el.GetType());
      FlatArray<BaseMappedIntegrationPoint*> fac_cents(3, lh);
      Array<BaseMappedIntegrationPoint*> other_cents;
      for (int i : Range(3))
      {
        FlatVector<const double> v1(2, et_verts[et_edges[i][0]]);
        FlatVector<const double> v2(2, et_verts[et_edges[i][1]]);
        fac_cents[i] = &trafo(IntegrationPoint(0.5*(v1+v2) | lh, 0), lh);

        Array<int> elnums;
        ma->GetFacetElements(facets[i], elnums);

        for (auto facel : elnums)
        {
          if (facel != el.Nr())
          {
            const auto &other_trafo = ma->GetTrafo(ElementId(facel), lh);
            // cout << "cc " << other_trafo(IntegrationPoint(1.0/3, 1.0/3), lh).GetPoint() << endl;
            other_cents.Append(&other_trafo(IntegrationPoint(1.0/3, 1.0/3), lh));
          }
        }
      }

      double uT = p1gf->Evaluate(xT_mip);
      bool limited = false;
      if (other_cents.Size() == 3)
      {
        FlatVector<> uTi(3, lh);
        for (int i : Range(3))
          uTi[i] = p1gf->Evaluate(*other_cents[i]);

        FlatVector<> delta(3, lh);
        for (int i : Range(3))
        {
          FlatMatrix<> lmat(2, 2, lh);
          lmat.Col(0) = other_cents[i]->GetPoint() - xT;
          lmat.Col(1) = other_cents[(i+1)%3]->GetPoint() - xT;
          CalcInverse(lmat);
          FlatVector<> lam = lmat * (fac_cents[i]->GetPoint() - xT) | lh;

          auto delta_u = lam[0]*(uTi[i]-uT) + lam[1]*(uTi[(i+1)%3]-uT);

          auto orig_val = p1gf->Evaluate(*fac_cents[i]) - uT;
          delta[i] = minmod_TVB(orig_val, theta*delta_u, M, h);
          if (delta[i] != orig_val)
            limited = true;
        }

        if (limited)
        {
          double neg_sum = 0, pos_sum = 0;
          for (auto d : delta)
          {
            neg_sum += negPart(d);
            pos_sum += posPart(d);
          }

          if (pos_sum - neg_sum != 0)
          {
            for (int i : Range(3))
              delta[i] = min(1.0, neg_sum/pos_sum)*posPart(delta[i]) - min(1.0, pos_sum/neg_sum)*negPart(delta[i]);
          }

          for (int i : Range(3))
            delta[i] += uT;

          IntegrationRule ir(el.GetType(), 1+fel.Order());
          auto &mir = trafo(ir, lh);

          ScalarFE<ET_TRIG,1> trig;
          FlatMatrix<> lag_shape(3, ir.GetNIP(), lh);
          IntegrationRule ir_facet(ir.GetNIP(), lh);
          for (int i : Range(ir))
          {
            ir_facet[i].Point()[0] = 1-2*ir[i].Point()[1];
            ir_facet[i].Point()[1] = 1-2*ir[i].Point()[0];
          }
          trig.CalcShape(ir_facet, lag_shape);
          project(el, mir, lag_shape, delta, u, lh);
        }
      }

      if (nonneg)
      {
        if (uT < 0)
        {
          cout << IM(2) << "Average is negative on el " << el.Nr() << ", setting to zero." << endl;
          FlatVector<> zeros(fel.GetNDof(), lh);
          zeros = 0;
          u->SetElementVector(el.GetDofs(), zeros);
          return;
        }

        IntegrationRule nnir(el.GetType(), fel.Order());
        FlatVector<> nnvec(nnir.GetNIP(), lh);
        u->Evaluate(trafo(nnir, lh), nnvec.AsMatrix(nnir.GetNIP(), 1));
        auto negative = any_of(nnvec.begin(), nnvec.end(), (bool(*)(double))signbit);

        if (negative)
        {
          FlatVector<double> vvals(3, lh);
          if (limited)
          {
            for (int i : Range(3))
              vvals[i] = u->Evaluate(trafo(IntegrationPoint(et_verts[i][0], et_verts[i][1]), lh));

          }
          else
          {
            for (int i : Range(3))
              vvals[i] = p1gf->Evaluate(trafo(IntegrationPoint(et_verts[i][0], et_verts[i][1]), lh));
          }
          auto minval = *min_element(vvals.begin(), vvals.end());

          if (minval == uT)
            return; // all vals >= 0, because we already checked uT >= 0

          double s = uT/(uT-minval);
          for (int i : Range(3))
            vvals[i] =  s*(vvals[i] - uT) + uT;

          IntegrationRule ir(el.GetType(), 1+fel.Order());
          auto &mir = trafo(ir, lh);
          ScalarFE<ET_TRIG,1> trig;
          FlatMatrix<> lag_shape(3, ir.GetNIP(), lh);
          trig.CalcShape(ir, lag_shape);
          project(el, mir, lag_shape, vvals, u, lh);

        }
      }

    });
}
