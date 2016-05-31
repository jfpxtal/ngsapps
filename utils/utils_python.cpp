#ifdef NGS_PYTHON

#include <python_ngstd.hpp>
#include "randomcf.hpp"

using namespace ngfem;

void ExportNgsAppsUtils() 
{
  cout << "exporting ngsapps.utils"  << endl;

  // Export RandomCoefficientFunction to python (name "RandomCF")

  bp::class_<RandomCoefficientFunction, shared_ptr<RandomCoefficientFunction>, bp::bases<CoefficientFunction>, boost::noncopyable>("RandomCF", bp::no_init)
    .def("__init__", bp::make_constructor 
         (FunctionPointer ([](double lower, double upper)
                           {
                             return make_shared<RandomCoefficientFunction> (lower, upper);
                           }),
          bp::default_call_policies(),        // need it to use named arguments
          (bp::arg("lower_bound")=0.0,bp::arg("upper_bound")=1.0)
           ));
  
  REGISTER_PTR_TO_PYTHON_BOOST_1_60_FIX(shared_ptr<RandomCoefficientFunction>);

  bp::implicitly_convertible 
    <shared_ptr<RandomCoefficientFunction>, shared_ptr<CoefficientFunction> >(); 
  
}

BOOST_PYTHON_MODULE(libngsapps_utils) 
{
  ExportNgsAppsUtils();
}
#endif // NGSX_PYTHON
