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

void limit(shared_ptr<GridFunction> u, double theta, double M, double h)
{
  const auto fes = u->GetFESpace();
  const auto ma = fes->GetMeshAccess();
  // Flags flags;
  // flags.SetFlag("novisual");
  // auto res = CreateGridFunction(fes, "gfu", flags);
  // res->Update();
  // res->GetVector() = u->GetVector();
  auto cpvec = u->GetVector().CreateVector();
  cpvec = u->GetVector();
  const auto diffop = fes->GetEvaluator(VOL);
  auto single_evaluator =  fes->GetEvaluator(VOL);
  if (dynamic_pointer_cast<BlockDifferentialOperator>(single_evaluator))
    single_evaluator = dynamic_pointer_cast<BlockDifferentialOperator>(single_evaluator)->BaseDiffOp();

  auto trial = make_shared<ProxyFunction>(false, false, single_evaluator,
                                          nullptr, nullptr, nullptr, nullptr, nullptr);
  auto test  = make_shared<ProxyFunction>(true, false, single_evaluator,
                                          nullptr, nullptr, nullptr, nullptr, nullptr);
  auto bli = make_shared<SymbolicBilinearFormIntegrator> (InnerProduct(trial,test), VOL, false);

  LocalHeap glh(100000, "limiter lh");
  IterateElements(*fes, VOL, glh, [&] (FESpace::Element el, LocalHeap &lh)
    {
      auto &trafo = el.GetTrafo();
      // cout << "l " << trafo(IntegrationPoint(0.0, 0.5), lh).GetPoint() << endl;
      // cout << "l " << trafo(IntegrationPoint(0.5, 0), lh).GetPoint() << endl;
      // cout << "l " << trafo(IntegrationPoint(0.5, 0.5), lh).GetPoint() << endl;
      // cout << "midp " << trafo(IntegrationPoint(1.0/3, 1.0/3), lh).GetPoint() << endl;

      // auto &xF1 = trafo(IntegrationPoint(0.0, 0.5), lh);
      // auto &xF2 = trafo(IntegrationPoint(0.5, 0), lh);
      // auto &xF3 = trafo(IntegrationPoint(0.5, 0.5), lh);
      auto &xT_mip = trafo(IntegrationPoint(1.0/3, 1.0/3), lh);
      auto xT = xT_mip.GetPoint();

      const auto &facets = el.Facets();
      const auto &et_edges = ElementTopology::GetEdges(el.GetType());
      const auto &et_verts = ElementTopology::GetVertices(el.GetType());
      FlatArray<BaseMappedIntegrationPoint*> fac_cents(3, lh);
      Array<BaseMappedIntegrationPoint*> other_cents;
      for (int i : Range(3))
      {
        // Array<int> pnums;
        // ma->GetFacetPNums(facets[i], pnums);
        // fac_cents.Row(i) = 0.5*(ma->GetPoint<2>(pnums[0]) + ma->GetPoint<2>(pnums[1]));
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

        // cout << "fac " << facets[i] << " els " << elnums << endl;
      }

      // cout << "fac cents " << fac_cents << endl;

      if (other_cents.Size() < 3)
      {
        cout << "Less than 3 neighbours for element " << el.Nr() << endl;
        return;
      }

      double uT = u->Evaluate(xT_mip);
      FlatVector<> uTi(3, lh);
      for (int i : Range(3))
        uTi[i] = u->Evaluate(*other_cents[i]);

      FlatVector<> delta(3, lh);
      for (int i : Range(3))
      {
        FlatMatrix<> lmat(2, 2, lh);
        lmat.Col(0) = other_cents[i]->GetPoint() - xT;
        lmat.Col(1) = other_cents[(i+1)%3]->GetPoint() - xT;
        CalcInverse(lmat);
        FlatVector<> lam = lmat * (fac_cents[i]->GetPoint() - xT) | lh;
        // cout << "lam " << lam;
        // cout << "xF1-xT " << fac_cents[i]->GetPoint()-xT << endl;
        // cout << "lc " << lam[0]*(other_cents[i]->GetPoint()-xT) + lam[1]*(other_cents[(i+1)%3]->GetPoint()-xT) << endl;

        auto delta_u = lam[0]*(uTi[i]-uT) + lam[1]*(uTi[(i+1)%3]-uT);
        // cout << "ex " << u->Evaluate(*fac_cents[i]) - uT << endl;
        // cout << "delta u " << delta_u << endl << endl;

        delta[i] = minmod_TVB(u->Evaluate(*fac_cents[i]) - uT, theta*delta_u, M, h);
        if (delta[i] != u->Evaluate(*fac_cents[i]) - uT)
          cout << "LIMITING el " << el.Nr() << "!!!!!!!!!!!!!!!!!!" << endl;
      }

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


      const auto &fel = el.GetFE();
      IntegrationRule ir(el.GetType(), 2*fel.Order());
      // FlatMatrix<> mfluxi(ir.GetNIP(), dimflux, lh);

      auto &mir = trafo(ir, lh);



      // coef->Evaluate (mir, mfluxi);

      // for (int j : Range(ir))
      //   mfluxi.Row(j) *= mir[j].GetWeight();


      ScalarFE<ET_TRIG,1> trig;
      FlatMatrix<> lag_shape(3, ir.GetNIP(), lh);
      IntegrationRule ir_facet(ir.GetNIP(), lh);
      for (int i : Range(ir))
      {
        ir_facet[i].Point()[0] = 1-2*ir[i].Point()[1];
        ir_facet[i].Point()[1] = 1-2*ir[i].Point()[0];
      }
      trig.CalcShape(ir_facet, lag_shape);
      FlatMatrix<> comb_shape = Trans(lag_shape)*delta | lh;
      // cout << "lag_shape " << lag_shape << endl;
      // cout << "comb " << comb_shape << endl;

      for (int i : Range(ir))
        comb_shape(i) = mir[i].GetWeight()*(comb_shape(i) + uT);

      FlatVector<> l2_scals(fel.GetNDof(), lh);
      diffop->ApplyTrans(fel, mir, comb_shape, l2_scals, lh);
      // cout << "atransir " << l2_scals << endl;

      FlatMatrix<> elmat(fel.GetNDof(), lh);
      bli->CalcElementMatrix(fel, trafo, elmat, lh);
      FlatCholeskyFactors<double> invelmat(elmat, lh);

      FlatVector<> new_coefs(fel.GetNDof(), lh);
      invelmat.Mult(l2_scals, new_coefs);

      cpvec.SetIndirect(el.GetDofs(), new_coefs);
      // cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
      // cout << delta << endl;
      // cout << res->Evaluate(*fac_cents[0])-uT << endl;
      // cout << res->Evaluate(*fac_cents[1])-uT << endl;
      // cout << res->Evaluate(*fac_cents[2])-uT << endl;
    });
  u->GetVector() = cpvec;
}
