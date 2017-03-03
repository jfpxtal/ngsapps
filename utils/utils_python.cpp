#ifdef NGS_PYTHON

#include <python_ngstd.hpp>
#include "myvtkoutput.hpp"
#include "randomcf.hpp"
#include "composecf.hpp"
#include "convolutioncf.hpp"
#include "parameterlf.hpp"
#include "cachecf.hpp"
#include "zlogzcf.hpp"
#include "annulusspeedcf.hpp"

using namespace ngfem;

typedef PyWrapper<CoefficientFunction> PyCF;
typedef PyWrapper<ngcomp::FESpace> PyFES;
typedef PyWrapper<BilinearFormIntegrator> PyBFI;
PyCF MakeCoefficient (py::object val);

void ExportNgsAppsUtils(py::module &m)
{
  cout << "exporting ngsapps.utils"  << endl;

  // Export RandomCoefficientFunction to python (name "RandomCF")
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

  typedef PyWrapperDerived<ZLogZCoefficientFunction,CoefficientFunction> PyZLogZ;
  py::class_<PyZLogZ, PyCF>
    (m, "ZLogZCF")
    .def("__init__",
         [](PyZLogZ *instance, py::object cf)
         {
           new (instance) PyZLogZ(make_shared<ZLogZCoefficientFunction>(MakeCoefficient(cf).Get()));
         }
      );

  typedef PyWrapperDerived<AnnulusSpeedCoefficientFunction, CoefficientFunction> PyAnnulusSpeedCF;
  py::class_<PyAnnulusSpeedCF,PyCF>
    (m, "AnnulusSpeedCF", "")
    .def("__init__",
         [](PyAnnulusSpeedCF *instance, double Rinner, double Router, double phi0, double vout, double v0)
         {
           new (instance) PyAnnulusSpeedCF(make_shared<AnnulusSpeedCoefficientFunction>(Rinner, Router, phi0, vout, v0));
         },
         py::arg("Rinner"), py::arg("Router"), py::arg("phi0"), py::arg("vout"), py::arg("v0")
      )
    .def("Dx", [](PyAnnulusSpeedCF & self) -> PyCF
         {
           return self->Dx();
         })
    .def("Dy", [](PyAnnulusSpeedCF & self) -> PyCF
         {
           return self->Dy();
         })
    ;

  typedef PyWrapperDerived<ComposeCoefficientFunction, CoefficientFunction> PyComposeCF;
  py::class_<PyComposeCF,PyCF>
    (m, "Compose", "compose two coefficient functions, c2 after c1")
    .def ("__init__",
          [] (PyComposeCF *instance, py::object c1, py::object c2, shared_ptr<ngcomp::MeshAccess> ma)
          {
            new (instance) PyComposeCF(make_shared<ComposeCoefficientFunction>(MakeCoefficient(c1).Get(), MakeCoefficient(c2).Get(), ma));
          },
          py::arg("innercf"), py::arg("outercf"), py::arg("mesh")
      );

  typedef PyWrapperDerived<ParameterLFProxy, CoefficientFunction> PyParameterLFProxy;
  py::class_<PyParameterLFProxy,PyCF>
    (m, "ParameterLFProxy", "xPar, yPar, zPar coordinates for ParameterLF")
    .def("__init__",
         [](PyParameterLFProxy *instance, int direction)
         {
           new (instance) PyParameterLFProxy(make_shared<ParameterLFProxy>(direction));
         })
    ;

  typedef PyWrapperDerived<CompactlySupportedKernel, CoefficientFunction> PyCompactlySupportedKernel;
  py::class_<PyCompactlySupportedKernel,PyCF>
    (m, "CompactlySupportedKernel", "")
    .def("__init__",
         [](PyCompactlySupportedKernel *instance, double radius, double scale)
         {
           new (instance) PyCompactlySupportedKernel(make_shared<CompactlySupportedKernel>(radius, scale));
         },
         py::arg("radius"), py::arg("scale")=1.0
      );

  typedef PyWrapper<ngcomp::GridFunction> PyGF;
  typedef PyWrapperDerived<ParameterLinearFormCF, CoefficientFunction> PyParameterLF;
  py::class_<PyParameterLF,PyCF>
    (m, "ParameterLF",
      "Parameterized LinearForm\n"
      "This coefficient function calculates the value of the parameterized integral\n"
      "I(xPar, yPar, zPar) = \\int integrand(x, y, z, xPar, yPar, zPar) d(x, y, z)\n"
      "gf is a GridFunction\n"
      "integrand is a CoefficientFunction which linearly contains a TestFunction from the FESpace to which gf belongs.\n"
      "When calculating the integral, the test function is then replaced by gf.")
    .def ("__init__",
          [] (PyParameterLF *instance, py::object integrand, PyGF gf, int order, int repeat, vector<double> patchSize)
          {
            new (instance) PyParameterLF(make_shared<ParameterLinearFormCF>(MakeCoefficient(integrand).Get(), gf.Get(), order, repeat, patchSize));
          },
          py::arg("integrand"), py::arg("gf"), py::arg("order")=5, py::arg("repeat")=0, py::arg("patchSize")=vector<int>()
      )
    ;

  typedef PyWrapperDerived<ConvolutionCoefficientFunction, CoefficientFunction> PyConvolveCF;
  py::class_<PyConvolveCF,PyCF>
    (m, "ConvolveCF",
      "convolution of a general coefficient function with a coefficient function representing a kernel\n"
      "to calculate repeated convolutions of GridFunctions, use ParameterLF")
    .def ("__init__",
          [] (PyConvolveCF *instance, py::object cf, py::object kernel, shared_ptr<ngcomp::MeshAccess> ma, int order)
          {
            new (instance) PyConvolveCF(make_shared<ConvolutionCoefficientFunction>(MakeCoefficient(cf).Get(), MakeCoefficient(kernel).Get(), ma, order));
          },
          py::arg("cf"), py::arg("kernel"), py::arg("mesh"), py::arg("order")=5
      )
    .def("CacheCF", [](PyConvolveCF & self)
         {
           self->CacheCF();
         })
    .def("ClearCFCache", [](PyConvolveCF & self)
         {
           self->ClearCFCache();
         })
    ;

  typedef PyWrapperDerived<CacheCoefficientFunction, CoefficientFunction> PyCacheCF;
  py::class_<PyCacheCF,PyCF>
    (m, "Cache", "cache results of a coefficient function")
    .def ("__init__",
          [] (PyCacheCF *instance, py::object c, shared_ptr<ngcomp::MeshAccess> ma)
          {
            new (instance) PyCacheCF(make_shared<CacheCoefficientFunction>(MakeCoefficient(c).Get(), ma));
          },
          py::arg("cf"), py::arg("mesh")
      )
    .def("Invalidate", [](PyCacheCF & self)
          {
            self->Invalidate();
          })
    .def("Refresh", [](PyCacheCF & self)
         {
           self->Refresh();
         })
    ;

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
                             if (ma->GetDimension() == 1)
                               ret = make_shared<MyVTKOutput<1>> (ma, coefs, names, filename, subdivision, only_element, nocache);
                             else if (ma->GetDimension() == 2)
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
