#ifdef NGS_PYTHON

#include <python_ngstd.hpp>
#include "myvtkoutput.hpp"
#include "randomcf.hpp"

using namespace ngfem;

void ExportNgsAppsUtils()
{
  cout << "exporting ngsapps.utils"  << endl;

  // Export RandomCoefficientFunction to python (name "RandomCF")

  typedef PyWrapper<CoefficientFunction> PyCF;
  typedef PyWrapperDerived<RandomCoefficientFunction,CoefficientFunction> PyRCF;

  bp::class_<PyRCF, bp::bases<PyCF>>
    ("RandomCF", bp::no_init)
    .def("__init__", bp::make_constructor
         (FunctionPointer ([](double lower, double upper)
                           {
                             return new PyRCF(make_shared<RandomCoefficientFunction> (lower, upper));
                           }),
          bp::default_call_policies(),        // need it to use arguments
          (bp::arg("lower")=0.0, bp::arg("upper")=1.0))
      );
  

  REGISTER_PTR_TO_PYTHON_BOOST_1_60_FIX(shared_ptr<RandomCoefficientFunction>);

  using namespace ngcomp;

  REGISTER_PTR_TO_PYTHON_BOOST_1_60_FIX(shared_ptr<MyBaseVTKOutput>);

 
  bp::class_<MyBaseVTKOutput, shared_ptr<MyBaseVTKOutput>,  boost::noncopyable>("MyVTKOutput", bp::no_init)
    .def("__init__", bp::make_constructor
         (FunctionPointer ([](shared_ptr<MeshAccess> ma, bp::list coefs_list,
                              bp::list names_list, string filename, int subdivision,
                              int only_element, bool nocache)
                           {
                             Array<shared_ptr<CoefficientFunction> > coefs
                               = makeCArray<shared_ptr<CoefficientFunction>> (coefs_list);
                             Array<string > names
                               = makeCArray<string> (names_list);
                             shared_ptr<MyBaseVTKOutput> ret;
                             if (ma->GetDimension() == 2)
                               ret = make_shared<MyVTKOutput<2>>(ma, coefs, names, filename, subdivision, only_element, nocache);
                             else
                               ret = make_shared<MyVTKOutput<3>>(ma, coefs, names, filename, subdivision, only_element, nocache);
                             return ret;
                           }),

          bp::default_call_policies(),     // need it to use named arguments
          (
            bp::arg("ma"),
            bp::arg("coefs")= bp::list(),
            bp::arg("names") = bp::list(),
            bp::arg("filename") = "vtkout",
            bp::arg("subdivision") = 0,
            bp::arg("only_element") = -1,
            bp::arg("nocache") = false
            )
           )
      )

    .def("Do", FunctionPointer([](MyBaseVTKOutput & self, int heapsize)
                               {
                                 LocalHeap lh (heapsize, "VTKOutput-heap");
                                 self.Do(lh);
                               }),
         (bp::arg("self"),bp::arg("heapsize")=1000000))
    .def("Do", FunctionPointer([](MyBaseVTKOutput & self, const BitArray * drawelems, int heapsize)
                               {
                                 LocalHeap lh (heapsize, "VTKOutput-heap");
                                 self.Do(lh, drawelems);
                               }),
         (bp::arg("self"),bp::arg("drawelems"),bp::arg("heapsize")=1000000))
    .def("UpdateMesh", FunctionPointer([](MyBaseVTKOutput & self)
                                       {
                                         self.BuildGridString();
                                       }))

    ;
}

BOOST_PYTHON_MODULE(libngsapps_utils)
{
  ExportNgsAppsUtils();
}
#endif // NGSX_PYTHON
