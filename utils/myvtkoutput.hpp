#ifndef FILE_MYVTKOUTPUT_HPP
#define FILE_MYVTKOUTPUT_HPP

#include <comp.hpp>

namespace ngcomp
{
  class MyBaseVTKOutput
  {
  public:
    virtual void BuildGridString() = 0;
    virtual void Do(LocalHeap & lh, const BitArray * drawelems = 0) = 0;
  };

  template <int D>
  class MyVTKOutput : public MyBaseVTKOutput
  {
  protected:
    enum CellType
    {
      VTK_VERTEX           = 1,
      VTK_LINE             = 3,
      VTK_TRIANGLE         = 5,
      VTK_QUAD             = 9,
      VTK_TETRA            = 10,
      VTK_HEXAHEDRON       = 12,
      VTK_WEDGE            = 13,
      VTK_PYRAMID          = 14,
      VTK_PENTAGONAL_PRISM = 15,
      VTK_HEXAGONAL_PRISM  = 16
    };

    shared_ptr<MeshAccess> ma = nullptr;
    Array<shared_ptr<CoefficientFunction>> coefs;
    Array<string> fieldnames;
    string filename;
    string grid_str;
    int subdivision;
    int only_element = -1;

    Array<IntegrationPoint> ref_vertices;
    Array<INT<D+1>> ref_elements;
    Array<shared_ptr<ValueField>> value_field;

    int output_cnt = 0;

    shared_ptr<ofstream> fileout;

  public:

    MyVTKOutput(const Array<shared_ptr<CoefficientFunction>> &,
               const Flags &,shared_ptr<MeshAccess>);

    MyVTKOutput(shared_ptr<MeshAccess>, const Array<shared_ptr<CoefficientFunction>> &,
               const Array<string> &, string, int, int);

    static int ElementTypeToVTKType(int et);
    virtual void BuildGridString();
    void FillReferenceData();
    void PrintFieldData();

    virtual void Do(LocalHeap & lh, const BitArray * drawelems = 0);
  };

  class NumProcMyVTKOutput : public NumProc
  {
  protected:
    shared_ptr<MyBaseVTKOutput> vtkout = nullptr;
  public:
    NumProcMyVTKOutput (shared_ptr<PDE> apde, const Flags & flags);
    virtual ~NumProcMyVTKOutput() { }

    virtual string GetClassName () const
      {
        return "NumProcMyVTKOutput";
      }

    virtual void Do (LocalHeap & lh);
  };

}

#endif
