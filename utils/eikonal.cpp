#include "eikonal.hpp"

using namespace ngcomp;

void solveEikonal1D(shared_ptr<CoefficientFunction> rhs, shared_ptr<GridFunction> res)
{
  auto fes = res->GetFESpace();
  auto ir = IntegrationRule(ET_SEGM, 2*fes->GetOrder());
  LocalHeap lh(100000, "eikonal lh");
  vector<double> lvals, rvals;
  bool first = true;
  double lastp, lastv, rightbnd, sum = 0;
  int ne = 0;
  for (const auto &el : fes->Elements())
  {
    HeapReset hr(lh);
    ne++;
    const auto &trafo = el.GetTrafo();
    auto &mir = trafo(ir, lh);
    FlatVector<> rhsvals(ir.Size(), lh);
    rhs->Evaluate(mir, rhsvals.AsMatrix(ir.Size(), 1));
    rightbnd = trafo(IntegrationPoint(0), lh).GetPoint()[0];
    for (int i = mir.Size()-1; i >= 0; i--)
    {
      if (first)
      {
        lastp = trafo(IntegrationPoint(1), lh).GetPoint()[0];
        first = false;
      } else
        rvals.emplace_back((mir[i].GetPoint()[0]-lastp) * lastv);
      lastv = rhsvals[i];
      sum += (mir[i].GetPoint()[0]-lastp) * lastv;
      lvals.emplace_back(sum);
      lastp = mir[i].GetPoint()[0];
    }
  }
  rvals.emplace_back((rightbnd-lastp) * lastv);
  partial_sum(rvals.rbegin(), rvals.rend(), rvals.rbegin());
  Array<double> resvals(lvals.size());
  for (int j : Range(ne))
  {
    for (int i : Range(ir))
      resvals[j*ir.Size()+i] = min(lvals[(j+1)*ir.Size()-1-i], rvals[(j+1)*ir.Size()-1-i]);
  }

  auto ipcf = make_shared<IntegrationPointCoefficientFunction>(ne, ir.Size(), resvals);
  SetValues(ipcf, *res, VOL, nullptr, lh);
}

EikonalSolver2D::EikonalSolver2D(shared_ptr<FESpace> fes, const vector<Vec<2>> &refs)
  : ma(fes->GetMeshAccess()), distByPntByRef(refs.size()), anglesByEl(ma->GetNE())
{
  LocalHeap glh(10000, "eikonal 2d lh");
  IterateElements(*fes, VOL, glh, [&] (FESpace::Element el, LocalHeap &lh)
  // for (auto i : Range(ma->GetNE()))
    {
      if (el.GetType() != ET_TRIG)
        throw Exception("EikonalSolver2D cannot handle non-trig elements.");
      auto verts = el.Vertices();
      Vec<2> vert_cs[3];
      for (int i : Range(3)) vert_cs[i] = ma->GetPoint<2>(verts[i]);
      for (int i : Range(3))
      {
        Vec<2> e1 = vert_cs[(i+1)%3]-vert_cs[i];
        Vec<2> e2 = vert_cs[(i+2)%3]-vert_cs[i];
        auto angle = atan2(e2[1], e2[0])-atan2(e1[1], e1[0]);
        if (angle < 0) angle += 2*M_PI;
        if (angle > M_PI/2)
          cout << "Warning (EikonalSolver2D): Element " << el.Nr() << " has obtuse angle." << endl;
        anglesByEl[el.Nr()][i] = angle;
      }
  // }
    });
  for (auto i : Range(refs.size()))
  {
    for (auto j : Range(ma->GetNV()))
      distByPntByRef[i].emplace_back(make_pair(j, L2Norm(refs[i]-ma->GetPoint<2>(j))));
    sort(distByPntByRef[i].begin(), distByPntByRef[i].end(), [] (auto &a, auto &b) { return a.second < b.second; });
    // for (auto j : Range(ma->GetNV()))
    //   cout << distByPntByRef[i][j].first << " " << distByPntByRef[i][j].second << endl;
  }

  nodal_fes = CreateFESpace("nodal", ma, Flags().SetFlag("order", 1));
  nodal_gf = CreateGridFunction(nodal_fes, "Eikonal", Flags().SetFlag("novisual"));
  nodal_gf->Update();
}

void EikonalSolver2D::solve(shared_ptr<CoefficientFunction> rhs)
{
  LocalHeap glh(100000, "eikonal 2d lh");
  FlatVector<double> res(ma->GetNV(), glh);
  res = numeric_limits<double>::max(); //                            ok?
  ma->IterateElements(BND, glh, [&] (Ngs_Element el, LocalHeap &lh)
    {
      for (auto v : el.Vertices()) res[v] = 0;
    });

  vector<double> rhsvals(ma->GetNV());
  const auto &et_verts = ElementTopology::GetVertices(ET_TRIG);
  IntegrationRule ir(3, glh);
  for (auto i : Range(3)) ir[i] = IntegrationPoint(et_verts[i], 0);
  ma->IterateElements(VOL, glh, [&] (Ngs_Element el, LocalHeap &lh)
    {
      const auto &verts = el.Vertices();
      const auto &trafo = ma->GetTrafo(el, lh);
      const auto &mir = trafo(ir, lh);
      FlatVector<> vals(3, lh);
      rhs->Evaluate(mir, vals.AsMatrix(ir.Size(), 1));
      for (auto i : Range(3)) rhsvals[verts[i]] += vals[i];
    });

  for (auto i : Range(rhsvals.size()))
  {
    rhsvals[i] /= ma->GetVertexElements(i).Size();
    // cout << rhsvals[i] << endl;
  }

  bool again;
  do
  {
    for (auto &distByPnt : distByPntByRef)
    {
      for (auto step : {1, -1})
      {
        again = false;
        int start, end;
        if (step == 1)
        {
          start = 0;
          end = distByPnt.size();
        } else {
          start = distByPnt.size()-1;
          end = -1;
        }
        for (auto l=start; l != end; l += step)
        {
          HeapReset hr(glh);
          const auto &this_vert = distByPnt[l].first;
          auto els = ma->GetVertexElements(this_vert);
          for (const auto &elnr : els)
          {
            auto el = ma->GetElement(ElementId(VOL, elnr));
            auto verts = el.Vertices();
            Vec<2> vert_cs[3];
            for (int i : Range(3)) vert_cs[i] = ma->GetPoint<2>(verts[i]);
            double lengths[3];
            for (int i : Range(3))
            {
              Vec<2> e1 = vert_cs[(i+1)%3]-vert_cs[i];
              Vec<2> e2 = vert_cs[(i+2)%3]-vert_cs[i];
              lengths[i] = L2Norm(e1);
            }
            Array<int> sort_idxs(3, glh);
            int C_idx;
            for (auto i : Range(3))
            {
              if (verts[i] == this_vert)
              {
                C_idx = i;
                break;
              }
            }
            sort_idxs = Range(3);
            sort_idxs.RemoveElement(C_idx);
            if (res[verts[sort_idxs[0]]] > res[verts[sort_idxs[1]]])
              swap(sort_idxs[0], sort_idxs[1]);

            const auto &a = lengths[(sort_idxs[0]+1)%3];
            const auto &b = lengths[(sort_idxs[1]+1)%3];
            const auto &c = lengths[(C_idx+1)%3];
            const auto &alpha = anglesByEl[el.Nr()][sort_idxs[0]];
            const auto &beta = anglesByEl[el.Nr()][sort_idxs[1]];
            const auto &T_A = res[verts[sort_idxs[0]]];
            const auto &T_B = res[verts[sort_idxs[1]]];
            const auto &f_C = rhsvals[this_vert];
            auto T_C_old = res[this_vert];
            auto &T_C = res[this_vert];

            auto sintheta = (T_B-T_A) / (c*f_C);
            if (sintheta <= 1)
            {
              auto theta = asin(sintheta);
              if (max(0.0, alpha-M_PI/2) <= theta && theta <= M_PI/2 - beta)
              {
                auto h = a*sin(alpha-theta);
                T_C = min(T_C, h*f_C + T_B);
              } else {
                T_C = min({T_C, T_A+b*f_C, T_B+a*f_C});
              }
            } else {
              T_C = min({T_C, T_A+b*f_C, T_B+a*f_C});
            }
            if (abs(T_C-T_C_old)>1e-5) again = true;

            // nodal_gf->SetElementVector(Array<int>({this_vert}), Vector<double>({T_C}));
            // Ng_Redraw();
            // cout << "done " << this_vert << endl;
            // cin.ignore();

          } // for element
        } // for vertex
        if (!again) break;
      } // for orientation
      if (!again) break;
    } // for reference point
  } while (again);
  nodal_gf->SetElementVector(Array<int>(Range(ma->GetNV())), res);
}
