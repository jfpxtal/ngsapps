#include "lagrangefe.hpp"

namespace ngfem
{

LagrangeSegm::LagrangeSegm (int order)
  : ScalarFiniteElement<1> (order+1, order)
{ ; }


void LagrangeSegm::CalcShape (const IntegrationPoint & ip,
                                    BareSliceVector<> shape) const
{
  double x = ip(0);
  T_CalcShape (x, shape);
}

void LagrangeSegm::CalcDShape (const IntegrationPoint & ip,
                                  BareSliceMatrix<> dshape) const
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
      res *= (order * x - a) * (1.0 / (i - a));
  }

  return res;
}

template <class T>
void LagrangeSegm::T_CalcShape (const T &x, BareSliceVector<T> shape) const
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
    shape[ii++] = T_LagrangeBasis1D(order - i, x);
  }
}

LagrangeTrig :: LagrangeTrig (int order)
  : ScalarFiniteElement<2> ((order+1)*(order+2)/2, order)
{ ; }


void LagrangeTrig :: CalcShape (const IntegrationPoint & ip,
                                    BareSliceVector<> shape) const
{
  double x = ip(0);
  double y = ip(1);
  T_CalcShape (x, y, shape);
}

void LagrangeTrig::CalcDShape (const IntegrationPoint & ip,
                                  BareSliceMatrix<> dshape) const
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
    res *= (order * x - a) * (1.0 / (i - a));
  for (int b = 0; b < j; ++b)
    res *= (order * y - b) * (1.0 / (j - b));
  for (int c = i + j + 1; c <= order; ++c)
    res *= (c - order * x - order * y) * (1.0 / (c - i - j));

  return res;
}

template <class T>
void LagrangeTrig::T_CalcShape (const T & x, const T & y,
                                    BareSliceVector<T> shape) const
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




LagrangeTet :: LagrangeTet (int order)
  : ScalarFiniteElement<3> ((order+1)*(order+2)*(order+3)/6, order)
{ ; }


void LagrangeTet :: CalcShape (const IntegrationPoint & ip,
                                    BareSliceVector<> shape) const
{
  double x = ip(0);
  double y = ip(1);
  double z = ip(2);
  T_CalcShape (x, y, z, shape);
}

void LagrangeTet::CalcDShape (const IntegrationPoint & ip,
                                  BareSliceMatrix<> dshape) const
{
  AutoDiff<3> adx (ip(0), 0);
  AutoDiff<3> ady (ip(1), 1);
  AutoDiff<3> adz (ip(2), 2);
  Vector<AutoDiff<3> > shapearray(ndof);
  T_CalcShape<AutoDiff<3>> (adx, ady, adz, shapearray);
  for (int i = 0; i < ndof; i++)
  {
    dshape(i, 0) = shapearray[i].DValue(0);
    dshape(i, 1) = shapearray[i].DValue(1);
    dshape(i, 2) = shapearray[i].DValue(2);
  }
}

/*   ^ |\
 *   | |  \
 *   j |    \
 *     |______\_
 *       i-->
 */
template <class T>
T LagrangeTet::T_LagrangeBasis3D(int i, int j, int k, const T &x, const T &y, const T &z) const
{
  T res = 1.0;
  for (int a = 0; a < i; ++a)
    res *= (order * x - a) * (1.0 / (i - a));
  for (int b = 0; b < j; ++b)
    res *= (order * y - b) * (1.0 / (j - b));
  for (int c = 0; c < k; ++c)
    res *= (order * z - c) * (1.0 / (k - c));
  for (int d = i + j + k + 1; d <= order; ++d)
    res *= (d - order * x - order * y - order * z) * (1.0 / (d - i - j - k));

  return res;
}

template <class T>
void LagrangeTet::T_CalcShape (const T & x, const T & y, const T & z,
                                    BareSliceVector<T> shape) const
{
  int ii = 0;

  // vertex dofs
  const POINT3D *vertices = ElementTopology::GetVertices(ET_TET);
  int nvert = ElementTopology::GetNVertices(ET_TET);
  for (int i = 0; i < nvert; ++i)
  {
    shape[ii++] = T_LagrangeBasis3D(order * vertices[i][0],
                                    order * vertices[i][1],
                                    order * vertices[i][2],
                                    x, y, z);
  }


  // edge dofs
  const EDGE *edges = ElementTopology::GetEdges(ET_TET);
  int nedges = ElementTopology::GetNEdges(ET_TET);
  for (int i = 0; i < nedges; i++)
  {
    int es = edges[i][0], ee = edges[i][1];
    if (vnums[es] > vnums[ee]) swap (es, ee);
    for (int k = 1; k < order; ++k)
    {
      shape[ii++] = T_LagrangeBasis3D(k * vertices[ee][0] + (order - k) * vertices[es][0],
                                      k * vertices[ee][1] + (order - k) * vertices[es][1],
                                      k * vertices[ee][2] + (order - k) * vertices[es][2],
                                      x, y, z);
    }
  }

  // face dofs
  const FACE *faces = ElementTopology::GetFaces(ET_TET);
  int nfaces = ElementTopology::GetNFaces(ET_TET);
  for (int i = 0; i < nfaces; i++)
  {
    int e1 = faces[i][0], e2 = faces[i][1], e3 = faces[i][2];
    if (vnums[e2] > vnums[e3]) swap (e2, e3); // e3 > e2
    if (vnums[e1] > vnums[e2]) swap (e1, e2); // e2 > e1 but now (new) e2 can be > e3 - e1 is smallest!
    if (vnums[e2] > vnums[e3]) swap (e2, e3); // e3 > e2 > e1

    for (int j = 1; j < order; ++j)
    {
      for (int i = 1; i < order - j; ++i)
      {
        shape[ii++] = T_LagrangeBasis3D(i * vertices[e1][0] + j * vertices[e2][0] + (order - j - i) * vertices[e3][0],
                                        i * vertices[e1][1] + j * vertices[e2][1] + (order - j - i) * vertices[e3][1],
                                        i * vertices[e1][2] + j * vertices[e2][2] + (order - j - i) * vertices[e3][2],
                                        x, y, z);
      }
    }
  }

  // inner dofs
  for (int k = 1; k < order; ++k)
  {
    for (int j = 1; j < order - k; ++j)
    {
      for (int i = 1; i < order - j - k; ++i)
      {
        shape[ii++] = T_LagrangeBasis3D(i, j, k, x, y, z);
      }
    }
  }
}



} // namespace ngfem
