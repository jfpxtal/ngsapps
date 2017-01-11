#include "lagrangefespace.hpp"

namespace ngcomp
{

LagrangeFESpace::LagrangeFESpace(shared_ptr<MeshAccess> ama, const Flags &flags)
  : FESpace(ama, flags)
{
  cout << "Constructor of LagrangeFESpace" << endl;

  order = int(flags.GetNumFlag ("order", 2));

  // needed to draw solution function
  if (ama->GetDimension() == 2)
  {
    evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<2>>>();
    flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpGradient<2>>>();
    evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpIdBoundary<2>>>();
    flux_evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpGradientBoundary<2>>>();
  }
  else
  {
    evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<3>>>();
    flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpGradient<3>>>();
    evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpIdBoundary<3>>>();
    flux_evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpGradientBoundary<3>>>();
  }
  integrator[VOL] = GetIntegrators().CreateBFI("mass", ma->GetDimension(),
                                          make_shared<ConstantCoefficientFunction>(1));
  integrator[BND] = CreateBFI("robin", ma->GetDimension(), 
                                  make_shared<ConstantCoefficientFunction>(1));
}

LagrangeFESpace::~LagrangeFESpace()
{
  // nothing to do
}

void LagrangeFESpace::Update(LocalHeap & h)
{
  // some global update:

  int n_vert = ma->GetNV();
  int n_edge = ma->GetNEdges();
  int n_face = ma->GetNFaces();
  int n_cell = ma->GetNE();

  first_edge_dof.SetSize (n_edge+1);
  int ii = n_vert;
  for (int i = 0; i < n_edge; i++, ii+=order-1)
    first_edge_dof[i] = ii;
  first_edge_dof[n_edge] = ii;

  if (ma->GetDimension() == 3)
  {
    first_face_dof.SetSize (n_face+1);
    for (int i = 0; i < n_face; i++, ii+=(order-1)*(order-2)/2)
      first_face_dof[i] = ii;
    first_face_dof[n_face] = ii;
  }

  first_cell_dof.SetSize (n_cell+1);
  for (int i = 0; i < n_cell; i++, ii+=( ma->GetDimension() == 3 ? (order-1)*(order-2)*(order-3)/6 : (order-1)*(order-2)/2))
    first_cell_dof[i] = ii;
  first_cell_dof[n_cell] = ii;

  // cout << "first_edge_dof = " << endl << first_edge_dof << endl;
  // cout << "first_face_dof = " << endl << first_face_dof << endl;
  // cout << "first_cell_dof = " << endl << first_cell_dof << endl;

  ndof = ii;
}

void LagrangeFESpace::GetDofNrs(ElementId ei, Array<int> &dnums) const
{
  if (ei.VB() == VOL)
  {
    // returns dofs of element number elnr

    dnums.SetSize(0);

    Ngs_Element ngel = ma->GetElement (ei.Nr());

    // vertex dofs
    for (int i = 0; i < ma->GetDimension() + 1; i++)
      dnums.Append (ngel.vertices[i]);

    // edge dofs
    for (int i = 0; i < (ma->GetDimension() == 3 ? 6 : 3); i++)
    {
      int first = first_edge_dof[ngel.edges[i]];
      int next  = first_edge_dof[ngel.edges[i]+1];
      for (int j = first; j < next; j++)
        dnums.Append (j);
    }

    // face dofs
    if (ma->GetDimension() == 3)
      for (int i = 0; i < 4; i++)
      {
        int fi = (ma->GetDimension() == 3 ? ngel.faces[i] : ei.Nr());
        int first = first_face_dof[fi];
        int next  = first_face_dof[fi+1];
        for (int j = first; j < next; j++)
          dnums.Append (j);
      }

    int first = first_cell_dof[ei.Nr()];
    int next  = first_cell_dof[ei.Nr()+1];
    for (int j = first; j < next; j++)
      dnums.Append (j);
    // cout << "dnums = " << dnums << endl;
  }
  else
  {
    dnums.SetSize(0);

    Ngs_Element ngel = ma->GetSElement (ei.Nr());

    // vertex dofs
    for (int i = 0; i < ma->GetDimension(); i++)
      dnums.Append (ngel.vertices[i]);

    for (int i = 0; i < (ma->GetDimension() == 3 ? 3 : 1); ++i)
    {
      // edge dofs
      int first = first_edge_dof[ngel.edges[i]];
      int next  = first_edge_dof[ngel.edges[i]+1];
      for (int j = first; j < next; j++)
        dnums.Append (j);
    }

    if (ma->GetDimension() == 3)
    {
      // face dofs
      int first = first_face_dof[ngel.faces[0]];
      int next  = first_face_dof[ngel.faces[0]+1];
      for (int j = first; j < next; j++)
        dnums.Append (j);
    }
  }
}



// returns the reference element
const FiniteElement &LagrangeFESpace::GetFE(int elnr, LocalHeap &lh) const
{
  if (ma->GetDimension() == 2)
  {
    LagrangeTrig * trig = new (lh) LagrangeTrig(order);

    Ngs_Element ngel = ma->GetElement (elnr);

    for (int i = 0; i < 3; i++)
      trig->SetVertexNumber (i, ngel.vertices[i]);

    return *trig;
  }
  else
  {
    LagrangeTet * tet = new (lh) LagrangeTet(order);

    Ngs_Element ngel = ma->GetElement (elnr);

    for (int i = 0; i < 4; i++)
      tet->SetVertexNumber (i, ngel.vertices[i]);

    return *tet;
  }

}


// the same for the surface elements
const FiniteElement & LagrangeFESpace::GetSFE (int elnr, LocalHeap &lh) const
{
  if (ma->GetDimension() == 2)
  {
    LagrangeSegm * segm = new (lh) LagrangeSegm(order);

    Ngs_Element ngel = ma->GetSElement (elnr);

    for (int i = 0; i < 2; i++)
      segm->SetVertexNumber (i, ngel.vertices[i]);

    return *segm;
  }
  else
  {
    LagrangeTrig * trig = new (lh) LagrangeTrig(order);

    Ngs_Element ngel = ma->GetSElement (elnr);

    for (int i = 0; i < 3; i++)
      trig->SetVertexNumber (i, ngel.vertices[i]);

    return *trig;
  }
}

void LagrangeFESpace :: GetVertexDofNrs (int vnr, Array<int> & dnums) const
{
  dnums.SetSize(1);
  dnums[0] = vnr;
}
  
  
void LagrangeFESpace :: GetEdgeDofNrs (int ednr, Array<int> & dnums) const
{
  dnums = IntRange(first_edge_dof[ednr],first_edge_dof[ednr+1]);
}

void LagrangeFESpace :: GetFaceDofNrs (int fanr, Array<int> & dnums) const
{
  dnums.SetSize0();
  if (ma->GetDimension() < 3) return;
  dnums = IntRange(first_face_dof[fanr],first_face_dof[fanr+1]);
}

void LagrangeFESpace :: GetInnerDofNrs (int elnr, Array<int> & dnums) const
{
  dnums = IntRange(first_cell_dof[elnr],first_cell_dof[elnr+1]);
}


static RegisterFESpace<LagrangeFESpace> initifes("lagrangefespace");

} // namespace ngcomp
