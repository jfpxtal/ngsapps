#include "lagrangefe.hpp"

namespace ngfem
{

LagrangeSegm::LagrangeSegm (int order)
  : ScalarFiniteElement<1> (order+1, order)
{ ; }


void LagrangeSegm::CalcShape (const IntegrationPoint & ip,
                                    SliceVector<> shape) const
{
  double x = ip(0);
  T_CalcShape (x, shape);
}

void LagrangeSegm::CalcDShape (const IntegrationPoint & ip,
                                  SliceMatrix<> dshape) const
{
  AutoDiff<1> adx (ip(0), 0);
  Vector<AutoDiff<1> > shapearray(ndof);
  T_CalcShape<AutoDiff<1>> (adx, shapearray);
  for (int i = 0; i < ndof; i++)
    dshape(i, 0) = shapearray[i].DValue(0);
}

template <class T>
T LagrangeSegm::T_LagrangeBasis1D(int i, const T &x) const
{
  T res = 1.0;
  for (int a = 0; a <= order; ++a)
  {
    if (a != i)
      res *= (order * x - a) / (i - a);
  }

  return res;
}

template <class T>
void LagrangeSegm::T_CalcShape (const T &x, SliceVector<T> shape) const
{
  int ii = 0;

  // vertex dofs
  const POINT3D *vertices = ElementTopology::GetVertices(ET_SEGM);
  int nvert = ElementTopology::GetNVertices(ET_SEGM);
  for (int i = 0; i < nvert; ++i)
  {
    shape[ii++] = T_LagrangeBasis1D(order * vertices[i][0], x);
  }

  // inner dofs
  for (int i = 1; i < order; ++i)
  {
    shape[ii++] = T_LagrangeBasis1D(i, x);
  }
}

LagrangeTrig :: LagrangeTrig (int order)
  : ScalarFiniteElement<2> ((order+1)*(order+2)/2, order)
{ ; }


void LagrangeTrig :: CalcShape (const IntegrationPoint & ip,
                                    SliceVector<> shape) const
{
  double x = ip(0);
  double y = ip(1);
  T_CalcShape (x, y, shape);
}

void LagrangeTrig::CalcDShape (const IntegrationPoint & ip,
                                  SliceMatrix<> dshape) const
{
  AutoDiff<2> adx (ip(0), 0);
  AutoDiff<2> ady (ip(1), 1);
  Vector<AutoDiff<2> > shapearray(ndof);
  T_CalcShape<AutoDiff<2>> (adx, ady, shapearray);
  for (int i = 0; i < ndof; i++)
  {
    dshape(i, 0) = shapearray[i].DValue(0);
    dshape(i, 1) = shapearray[i].DValue(1);
  }
}

/*   ^ |\
 *   | |  \
 *   j |    \
 *     |______\_
 *       i-->
 */
template <class T>
T LagrangeTrig::T_LagrangeBasis2D(int i, int j, const T &x, const T &y) const
{
  T res = 1.0;
  for (int a = 0; a < i; ++a)
    res *= (order * x - a) / (i - a);
  for (int b = 0; b < j; ++b)
    res *= (order * y - b) / (j - b);
  for (int c = i + j + 1; c <= order; ++c)
    res *= (c - order * x - order * y) / (c - i - j);

  return res;
}

template <class T>
void LagrangeTrig::T_CalcShape (const T & x, const T & y,
                                    SliceVector<T> shape) const
{
  int ii = 0;

  // vertex dofs
  const POINT3D *vertices = ElementTopology::GetVertices(ET_TRIG);
  int nvert = ElementTopology::GetNVertices(ET_TRIG);
  for (int i = 0; i < nvert; ++i)
  {
    shape[ii++] = T_LagrangeBasis2D(order * vertices[i][0],
                                  order * vertices[i][1], x, y);
  }


  // edge dofs
  const EDGE *edges = ElementTopology::GetEdges(ET_TRIG);
  int nedges = ElementTopology::GetNEdges(ET_TRIG);
  for (int i = 0; i < nedges; i++)
  {
    int es = edges[i][0], ee = edges[i][1];
    if (vnums[es] > vnums[ee]) swap (es, ee);
    for (int k = 1; k < order; ++k)
    {
      shape[ii++] = T_LagrangeBasis2D(k * vertices[ee][0] + (order - k) * vertices[es][0],
                                    k * vertices[ee][1] + (order - k) * vertices[es][1],
                                    x, y);
    }
  }

  // inner dofs
  for (int j = 1; j < order; ++j)
  {
    for (int i = 1; i < order - j; ++i)
    {
      shape[ii++] = T_LagrangeBasis2D(i, j, x, y);
    }
  }
}

} // namespace ngfem
