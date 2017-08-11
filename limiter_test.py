from ngsolve import *
from ngsapps.utils import *
from netgen.geom2d import unit_square

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
# mesh = Mesh(unit_square.GenerateMesh(m))
Draw(mesh)
fes = L2(mesh, order=1)
g = GridFunction(fes)
h = GridFunction(fes)
g.Set(sin(1000*x*y))
h.vec.data = g.vec
# g.Set(CoefficientFunction(1))
Draw(h, mesh, 'h')
Limit(g, 1, 0.01, 1)
Draw(g, mesh, 'g')
print(Integrate(sqr(g-h), mesh))
