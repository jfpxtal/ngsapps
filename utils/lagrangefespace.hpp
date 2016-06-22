#ifndef FILE_LAGRANGEFESPACE_HPP
#define FILE_LAGRANGEFESPACE_HPP

#include <comp.hpp>
#include "lagrangefe.hpp"

namespace ngcomp
{

  class LagrangeFESpace : public FESpace
  {
    int order;
    int ndof;

    Array<int> first_edge_dof;
    Array<int> first_face_dof;
    Array<int> first_cell_dof;
  public:
    LagrangeFESpace(shared_ptr<MeshAccess> ama, const Flags & flags);

    virtual ~LagrangeFESpace();

    virtual string GetClassName() const
    {
      return "LagrangeFESpace";
    }

    virtual void Update(LocalHeap & lh);
    virtual int GetNDof() const { return ndof; }

    virtual void GetDofNrs(int elnr, Array<int> & dnums) const;
    virtual void GetSDofNrs(int selnr, Array<int> & dnums) const;

    virtual const FiniteElement & GetFE(int elnr, LocalHeap & lh) const;
    virtual const FiniteElement & GetSFE(int selnr, LocalHeap & lh) const;
  };

}

#endif
