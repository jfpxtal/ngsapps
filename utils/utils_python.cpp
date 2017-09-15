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
#include "limiter.hpp"
#include "eikonal.hpp"

using namespace ngfem;

typedef shared_ptr<CoefficientFunction> PyCF;
typedef shared_ptr<ngcomp::FESpace> PyFES;
typedef shared_ptr<BilinearFormIntegrator> PyBFI;
PyCF MakeCoefficient (py::object val);

void ExportNgsAppsUtils(py::module &m)
{
  cout << "exporting ngsapps.utils"  << endl;

  // Export RandomCoefficientFunction to python (name "RandomCF")
  typedef shared_ptr<RandomCoefficientFunction> PyRCF;
  py::class_<RandomCoefficientFunction, PyRCF, CoefficientFunction>
    (m, "RandomCF")
    .def("__init__",
         [](RandomCoefficientFunction *instance, double lower, double upper)
            {
              new (instance) RandomCoefficientFunction(lower, upper);
            },
          py::arg("lower")=0.0, py::arg("upper")=1.0
      );

  typedef shared_ptr<ZLogZCoefficientFunction> PyZLogZ;
  py::class_<ZLogZCoefficientFunction, PyZLogZ, CoefficientFunction>
    (m, "ZLogZCF")
    .def("__init__",
         [](ZLogZCoefficientFunction *instance, py::object cf)
         {
           new (instance) ZLogZCoefficientFunction(MakeCoefficient(cf));
         }
      );

  typedef shared_ptr<AnnulusSpeedCoefficientFunction> PyAnnulusSpeedCF;
  py::class_<AnnulusSpeedCoefficientFunction, PyAnnulusSpeedCF, CoefficientFunction>
    (m, "AnnulusSpeedCF", "")
    .def("__init__",
         [](AnnulusSpeedCoefficientFunction *instance, double Rinner, double Router, double phi0, double vout, double v0, double smearR, double smearphi)
         {
           new (instance) AnnulusSpeedCoefficientFunction(Rinner, Router, phi0, vout, v0, smearR, smearphi);
         },
         py::arg("Rinner"), py::arg("Router"), py::arg("phi0"), py::arg("vout"), py::arg("v0"), py::arg("smearR"), py::arg("smearphi")
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

  typedef shared_ptr<ComposeCoefficientFunction> PyComposeCF;
  py::class_<ComposeCoefficientFunction, PyComposeCF, CoefficientFunction>
    (m, "Compose", "compose two coefficient functions, c2 after c1")
    .def ("__init__",
          [] (ComposeCoefficientFunction *instance, py::object c1, py::object c2, shared_ptr<ngcomp::MeshAccess> ma)
          {
            new (instance) ComposeCoefficientFunction(MakeCoefficient(c1), MakeCoefficient(c2), ma);
          },
          py::arg("innercf"), py::arg("outercf"), py::arg("mesh")
      );

  typedef shared_ptr<ParameterLFProxy> PyParameterLFProxy;
  py::class_<ParameterLFProxy, PyParameterLFProxy, CoefficientFunction>
    (m, "ParameterLFProxy", "xPar, yPar, zPar coordinates for ParameterLF")
    .def("__init__",
         [](ParameterLFProxy *instance, int direction)
         {
           new (instance) ParameterLFProxy(direction);
         })
    ;

  typedef shared_ptr<CompactlySupportedKernel> PyCompactlySupportedKernel;
  py::class_<CompactlySupportedKernel, PyCompactlySupportedKernel, CoefficientFunction>
    (m, "CompactlySupportedKernel", "")
    .def("__init__",
         [](CompactlySupportedKernel *instance, double radius, double scale)
         {
           new (instance) CompactlySupportedKernel(radius, scale);
         },
         py::arg("radius"), py::arg("scale")=1.0
      );

  typedef shared_ptr<ParameterLinearFormCF> PyParameterLF;
  py::class_<ParameterLinearFormCF, PyParameterLF, CoefficientFunction>
    (m, "ParameterLF",
      "Parameterized LinearForm\n"
      "This coefficient function calculates the value of the parameterized integral\n"
      "I(xPar, yPar, zPar) = \\int integrand(x, y, z, xPar, yPar, zPar) d(x, y, z)\n"
      "gf is a GridFunction\n"
      "integrand is a CoefficientFunction which linearly contains a TestFunction from the FESpace to which gf belongs.\n"
      "When calculating the integral, the test function is then replaced by gf.")
    .def ("__init__",
          [] (ParameterLinearFormCF *instance, py::object integrand, shared_ptr<ngcomp::GridFunction> gf, int order, int repeat, vector<double> patchSize)
          {
            new (instance) ParameterLinearFormCF(MakeCoefficient(integrand), gf, order, repeat, patchSize);
          },
          py::arg("integrand"), py::arg("gf"), py::arg("order")=5, py::arg("repeat")=0, py::arg("patchSize")=vector<int>()
      )
    ;

  typedef shared_ptr<ConvolutionCoefficientFunction> PyConvolveCF;
  py::class_<ConvolutionCoefficientFunction, PyConvolveCF, CoefficientFunction>
    (m, "ConvolveCF",
      "convolution of a general coefficient function with a coefficient function representing a kernel\n"
      "to calculate repeated convolutions of GridFunctions, use ParameterLF")
    .def ("__init__",
          [] (ConvolutionCoefficientFunction *instance, py::object cf, py::object kernel, shared_ptr<ngcomp::MeshAccess> ma, int order)
          {
            new (instance) ConvolutionCoefficientFunction(MakeCoefficient(cf), MakeCoefficient(kernel), ma, order);
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

  typedef shared_ptr<CacheCoefficientFunction> PyCacheCF;
  py::class_<CacheCoefficientFunction, PyCacheCF, CoefficientFunction>
    (m, "Cache", "cache results of a coefficient function")
    .def ("__init__",
          [] (CacheCoefficientFunction *instance, py::object c, shared_ptr<ngcomp::MeshAccess> ma)
          {
            new (instance) CacheCoefficientFunction(MakeCoefficient(c), ma);
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

  typedef shared_ptr<MyBaseVTKOutput> PyMyVTK;
  m.def("MyVTKOutput",
         [](shared_ptr<MeshAccess> ma, py::list coefs_list,
            py::list names_list, string filename, int subdivision, int only_element, bool nocache) -> PyMyVTK
                           {
                             Array<shared_ptr<CoefficientFunction> > coefs
                               = makeCArraySharedPtr<shared_ptr<CoefficientFunction>> (coefs_list);
                             Array<string > names
                               = makeCArray<string> (names_list);
                             if (ma->GetDimension() == 1)
                               return make_shared<MyVTKOutput<1>> (ma, coefs, names, filename, subdivision, only_element, nocache);
                             else if (ma->GetDimension() == 2)
                               return make_shared<MyVTKOutput<2>> (ma, coefs, names, filename, subdivision, only_element, nocache);
                             else
                               return make_shared<MyVTKOutput<3>> (ma, coefs, names, filename, subdivision, only_element, nocache);
                           },

            py::arg("ma"),
            py::arg("coefs")= py::list(),
            py::arg("names") = py::list(),
            py::arg("filename") = "vtkout",
            py::arg("subdivision") = 0,
            py::arg("only_element") = -1,
            py::arg("nocache") = false
        );

  py::class_<MyBaseVTKOutput, PyMyVTK>(m, "C_MyVTKOutput")
    .def("Do", [](PyMyVTK & self, int heapsize)
                               {
                                 LocalHeap lh (heapsize, "VTKOutput-heap");
                                 self->Do(lh);
                               },
         py::arg("heapsize")=1000000)
    .def("Do", [](PyMyVTK & self, const BitArray * drawelems, int heapsize)
                               {
                                 LocalHeap lh (heapsize, "VTKOutput-heap");
                                 self->Do(lh, drawelems);
                               },
         py::arg("drawelems"),py::arg("heapsize")=1000000)
    .def("UpdateMesh", [](PyMyVTK & self)
                               {
                                 self->BuildGridString();
                               })

    ;

  m.def("Project", &project);
  m.def("Limit", &limit);
  m.def("LimitOld", &limitold);
  m.def("CreateIPCF", [] (int elems, int size, vector<double> &vals) -> PyCF
        {
          auto res = make_shared<IntegrationPointCoefficientFunction>(elems, size);
          for (auto e : Range(elems))
          {
            for (auto i : Range(size))
              (*res)(e, i) = vals[e*size+i];
          }
          // return static_pointer_cast<CoefficientFunction>(res);
          return res;
        });
  m.def("SolveEikonal1D", &solveEikonal1D);
}

PYBIND11_PLUGIN(libngsapps_utils)
{
  py::module m("ngsapps", "pybind ngsapps");
  ExportNgsAppsUtils(m);
  return m.ptr();
}
#endif // NGSX_PYTHON
