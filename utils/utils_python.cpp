#ifdef NGS_PYTHON

#include <python_ngstd.hpp>
#include "myvtkoutput.hpp"
#include "randomcf.hpp"

using namespace ngfem;

void ExportNgsAppsUtils(py::module &m)
{
  cout << "exporting ngsapps.utils"  << endl;

  // Export RandomCoefficientFunction to python (name "RandomCF")

  typedef PyWrapper<CoefficientFunction> PyCF;
  typedef PyWrapperDerived<RandomCoefficientFunction,CoefficientFunction> PyRCF;

  py::class_<PyRCF, PyCF>
    (m, "RandomCF")
    .def("__init__",
         [](PyRCF *instance, double lower, double upper)
            {
              new (instance) PyRCF(make_shared<RandomCoefficientFunction> (lower, upper));
            },
          py::arg("lower")=0.0, py::arg("upper")=1.0
      );

  using namespace ngcomp;

  typedef PyWrapper<MyBaseVTKOutput> PyMyVTK;
  py::class_<PyMyVTK>(m, "MyVTKOutput")
    .def("__init__",
         [](PyMyVTK *instance, shared_ptr<MeshAccess> ma, py::list coefs_list,
            py::list names_list, string filename, int subdivision, int only_element, bool nocache)
                           {
                             Array<shared_ptr<CoefficientFunction> > coefs
                               = makeCArrayUnpackWrapper<PyCF> (coefs_list);
                             Array<string > names
                               = makeCArray<string> (names_list);
                             shared_ptr<MyBaseVTKOutput> ret;
                             if (ma->GetDimension() == 2)
                               ret = make_shared<MyVTKOutput<2>> (ma, coefs, names, filename, subdivision, only_element, nocache);
                             else
                               ret = make_shared<MyVTKOutput<3>> (ma, coefs, names, filename, subdivision, only_element, nocache);
                             new (instance) PyMyVTK(ret);
                           },

            py::arg("ma"),
            py::arg("coefs")= py::list(),
            py::arg("names") = py::list(),
            py::arg("filename") = "vtkout",
            py::arg("subdivision") = 0,
            py::arg("only_element") = -1,
            py::arg("nocache") = false
      )

    .def("Do", FunctionPointer([](PyMyVTK & self, int heapsize)
                               {
                                 LocalHeap lh (heapsize, "VTKOutput-heap");
                                 self->Do(lh);
                               }),
         py::arg("heapsize")=1000000)
    .def("Do", FunctionPointer([](PyMyVTK & self, const BitArray * drawelems, int heapsize)
                               {
                                 LocalHeap lh (heapsize, "VTKOutput-heap");
                                 self->Do(lh, drawelems);
                               }),
         py::arg("drawelems"),py::arg("heapsize")=1000000)
    .def("UpdateMesh", FunctionPointer([](PyMyVTK & self)
                               {
                                 self->BuildGridString();
                               }))

    ;
}

PYBIND11_PLUGIN(libngsapps_utils)
{
  py::module m("ngsapps", "pybind ngsapps");
  ExportNgsAppsUtils(m);
  return m.ptr();
}
#endif // NGSX_PYTHON
