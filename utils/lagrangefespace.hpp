#ifndef FILE_LAGRANGEFESPACE_HPP
#define FILE_LAGRANGEFESPACE_HPP

#include <comp.hpp>
#include "lagrangefe.hpp"

namespace ngcomp
{

  class LagrangeFESpace : public FESpace
  {
    int order;
    size_t ndof;

    Array<int> first_edge_dof;
    Array<int> first_face_dof;
    Array<int> first_cell_dof;
  public:
    LagrangeFESpace(shared_ptr<MeshAccess> ama, const Flags & flags);

    virtual ~LagrangeFESpace();

    virtual string GetClassName() const override
    {
      return "LagrangeFESpace";
    }

    virtual void Update(LocalHeap & lh) override;
    virtual size_t GetNDof() const throw() override { return ndof; }

    virtual void GetDofNrs(ElementId, Array<int> & dnums) const override;

    virtual FiniteElement & GetFE(ElementId ei, Allocator & lh) const override;

    virtual void GetVertexDofNrs (int vnr, Array<int> & dnums) const override;
    virtual void GetEdgeDofNrs (int ednr, Array<int> & dnums) const override;
    virtual void GetFaceDofNrs (int fanr, Array<int> & dnums) const override;
    virtual void GetInnerDofNrs (int elnr, Array<int> & dnums) const override;
  };

}

#endif
