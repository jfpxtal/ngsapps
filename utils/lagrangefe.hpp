#ifndef FILE_LAGRANGEFE_HPP
#define FILE_LAGRANGEFE_HPP

#include <fem.hpp>

namespace ngfem
{

  class LagrangeSegm : public ScalarFiniteElement<1>
  {
    int vnums[2];
  public:
    LagrangeSegm (int order);
    virtual ELEMENT_TYPE ElementType() const { return ET_SEGM; }
    void SetVertexNumber(int i, int v) { vnums[i] = v; }

    virtual void CalcShape(const IntegrationPoint & ip,
                            SliceVector<> shape) const;

    virtual void CalcDShape(const IntegrationPoint & ip,
                             SliceMatrix<> dshape) const;

  private:
    template <class T>
    T T_LagrangeBasis1D(int i, const T &x) const;
    template <class T>
    void T_CalcShape(const T &x, SliceVector<T> shape) const;
  };

  class LagrangeTrig : public ScalarFiniteElement<2>
  {
    int vnums[3];
  public:
    LagrangeTrig(int order);
    virtual ELEMENT_TYPE ElementType() const { return ET_TRIG; }
    void SetVertexNumber(int i, int v) { vnums[i] = v; }

    virtual void CalcShape(const IntegrationPoint & ip,
                            SliceVector<> shape) const;

    virtual void CalcDShape(const IntegrationPoint & ip,
                             SliceMatrix<> dshape) const;

  private:
    template <class T>
    T T_LagrangeBasis2D(int i, int j, const T &x, const T &y) const;
    template <class T>
    void T_CalcShape(const T & x, const T & y, SliceVector<T> shape) const;
  };

  class LagrangeTet : public ScalarFiniteElement<3>
  {
    int vnums[4];
  public:
    LagrangeTet(int order);
    virtual ELEMENT_TYPE ElementType() const { return ET_TET; }
    void SetVertexNumber(int i, int v) { vnums[i] = v; }

    virtual void CalcShape(const IntegrationPoint & ip,
                            SliceVector<> shape) const;

    virtual void CalcDShape(const IntegrationPoint & ip,
                             SliceMatrix<> dshape) const;

  private:
    template <class T>
    T T_LagrangeBasis3D(int i, int j,  int k, const T &x, const T &y, const T &z) const;
    template <class T>
    void T_CalcShape(const T & x, const T & y, const T & z, SliceVector<T> shape) const;
  };

}

#endif

