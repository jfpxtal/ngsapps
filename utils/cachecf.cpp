#include "cachecf.hpp"

namespace ngbla {
  bool operator== (const Vector<double> &v1, const Vector<double> &v2)
  {
    if (v1.Size() != v2.Size()) return false;
    for (int i = 0; i < v1.Size(); i++)
      if (v1(i) != v2(i)) return false;
    return true;
  }

  bool operator< (const Vector<double> &v1, const Vector<double> &v2)
  {
    for (int i = 0; i < v1.Size(); i++)
    {
      if (v1(i) < v2(i)) return true;
      else if (v2(i) < v1(i)) return false;
    }
    return false;
  }
}

namespace ngfem
{

  CacheCoefficientFunction::CacheCoefficientFunction (shared_ptr<CoefficientFunction> ac, shared_ptr<ngcomp::MeshAccess> ama)
    : CoefficientFunction(ac->Dimension(), ac->IsComplex()), c(ac), ma(ama)
  {
    cache.max_load_factor(0.1);
  }

  void CacheCoefficientFunction::PrintReport (ostream & ost) const
  {
    ost << "Cache(";
    c->PrintReport(ost);
    ost << ")";
  }

  void CacheCoefficientFunction::TraverseTree (const function<void(CoefficientFunction&)> & func)
  {
    c->TraverseTree (func);
    func(*this);
  }

  double CacheCoefficientFunction::Evaluate (const BaseMappedIntegrationPoint & ip) const
  {
    auto point = ip.GetPoint();
    auto it = cache.find(point);
    if (it == cache.end())
    {
      // cache miss
      // cout << "MISS ";
      ValueType vt(c->Evaluate(ip), ip.IP(), ip.GetTransformation().GetElementId());
      auto noconst = const_cast<CacheCoefficientFunction*>(this);
      std::lock_guard<std::mutex> guard(noconst->cacheWrite);
      noconst->cache.insert(std::make_pair(Vector<double>(point), vt));
      return vt.value;
    } else {
      // cache hit
      return it->second.value;
    }
  }

  void CacheCoefficientFunction::Invalidate()
  {
    std::lock_guard<std::mutex> guard(cacheWrite);
    // should not deallocate reserved space
    cache.clear();
  }

  // std::map version
  // void CacheCoefficientFunction::Refresh()
  // {
  //   LocalHeap clh(100000, "cachecf refresh");
  //   if (task_manager)
  //   {
  //     task_manager -> CreateJob
  //       ( [&] (const TaskInfo & ti)
  //         {
  //           LocalHeap lh = clh.Split(ti.thread_nr, ti.nthreads);
  //           size_t cnt = 0;
  //           for (auto it = cache.begin(); it != cache.end(); ++it, cnt++)
  //           {
  //             if (cnt % ti.nthreads != ti.thread_nr) continue;

  //             HeapReset hr(lh);

  //             if (it->second.ip.Nr() == -666)
  //             {
  //               // dummy MappedIntegrationPoint used by convolutioncf
  //               it->second.value = c->Evaluate(DummyMIPFromPoint(it->first, lh));
  //             } else {
  //               auto &trafo = ma->GetTrafo(it->second.ei, lh);
  //               it->second.value = c->Evaluate(trafo(it->second.ip, lh));
  //             }
  //           }
  //         } );
  //   }
  // }

  void CacheCoefficientFunction::Refresh()
  {
    // ofstream ofile("cache.txt");
    LocalHeap clh(100000, "cachecf refresh");
    if (task_manager)
    {
      SharedLoop sl(cache.bucket_count());
      // ofile << endl << "BUCKET CNT " << cache.bucket_count() << endl;;

      task_manager -> CreateJob
        ( [&] (const TaskInfo & ti)
          {
            LocalHeap lh = clh.Split(ti.thread_nr, ti.nthreads);
            for (size_t mynr : sl)
            {
              auto it = cache.begin(mynr);
              auto end = cache.end(mynr);
              // int cnt = 0;
              while (it != end)
              {
                HeapReset hr(lh);

                // auto mip = it->second.eltrans(it->second.ip, lh);
                if (it->second.ip.Nr() == -666)
                {
                  // dummy MappedIntegrationPoint used by convolutioncf
                  it->second.value = c->Evaluate(DummyMIPFromPoint(it->first, lh));
                } else {
                  auto &trafo = ma->GetTrafo(it->second.ei, lh);
                  it->second.value = c->Evaluate(trafo(it->second.ip, lh));
                }
                // cnt++;
                ++it;
              }
              // ofile << cnt << " ";
            }
          } );
      // ofile << endl << endl;
    }
    // TODO: else
    // else
    // {
    //   for (auto i : Range(GetNE(vb)))
    //   {
    //     HeapReset hr(clh);
    //     ElementId ei(vb, i);
    //     Ngs_Element el(GetElement(ei), ei);
    //     func (move(el), clh);
    //   }
    // }
  }
}
