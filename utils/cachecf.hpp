#pragma once

#include <comp.hpp>
#include <python_ngstd.hpp>
#include <unordered_map>

#include "utils.hpp"

namespace ngbla {
  bool operator== (const Vector<double> &v1, const Vector<double> &v2);
  bool operator< (const Vector<double> &v1, const Vector<double> &v2);
}

// TODO: check sizeof(size_t)
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  std::hash<T> hasher;
  const std::size_t kMul = 0x9ddfea08eb382d69ULL;
  std::size_t a = (hasher(v) ^ seed) * kMul;
  a ^= (a >> 47);
  std::size_t b = (seed ^ a) * kMul;
  b ^= (b >> 47);
  seed = b * kMul;
}

namespace std {
  template<> struct hash<Vector<double>> {
    std::size_t operator()(const Vector<double> &v) const {
      std::size_t ret = 0;
      for (int i = 0; i < v.Size(); i++)
        hash_combine(ret, v(i));
      return ret;
    }
  };
}

namespace ngfem
{
  class CacheCoefficientFunction : public CoefficientFunction
  {
    struct ValueType
    {
      double value;
      IntegrationPoint ip;
      ElementId ei;

      ValueType(double aval, IntegrationPoint aip, ElementId aei) : value(aval), ip(aip), ei(aei) {}
    };

    shared_ptr<CoefficientFunction> c;
    shared_ptr<ngcomp::MeshAccess> ma;
    std::unordered_map<Vector<double>, ValueType> cache;
    // std::map<Vector<double>, ValueType> cache;

    std::mutex cacheWrite;

  public:
    CacheCoefficientFunction (shared_ptr<CoefficientFunction> ac, shared_ptr<ngcomp::MeshAccess> ama);
    virtual ~CacheCoefficientFunction () {}
    ///
    virtual double Evaluate (const BaseMappedIntegrationPoint & ip) const;
    virtual void TraverseTree (const function<void(CoefficientFunction&)> & func);
    virtual void PrintReport (ostream & ost) const;
    virtual void Invalidate();
    virtual void Refresh();
  };
}

