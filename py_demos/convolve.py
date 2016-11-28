from ngsolve import *
from ngsapps.utils import *
from netgen.geom2d import unit_square, SplineGeometry

geo = SplineGeometry()
geo.AddRectangle((-5,-1), (5, 1))
mesh = Mesh(geo.GenerateMesh(maxh=0.5))
cf2 = CoefficientFunction(IfPos(x+1, IfPos(x-1, 0, 1), 0))
# cf2 = CoefficientFunction(IfPos(0.8-x, 0, 1))
# cf2 = cf1
# cf1 = CoefficientFunction(IfPos(x, exp(-2*x), 0))
# cf2 = CoefficientFunction(IfPos(x, IfPos(1-x, 1, 0), 0))
cf1 = CoefficientFunction(exp(-5*pow(x, 2)-5*pow(y, 2)))

conv = Convolve(cf1, cf2, mesh)
Draw(cf1, mesh, "cf1")
Draw(cf2, mesh, "cf2")
Draw(conv, mesh, "conv")
