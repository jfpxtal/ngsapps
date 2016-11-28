from ngsolve import *
from ngsapps.utils import *
from netgen.geom2d import unit_square, SplineGeometry

geo = SplineGeometry()
geo.AddRectangle((-5,-1), (5, 1))
mesh = Mesh(geo.GenerateMesh(maxh=0.5))

Draw(Compose(x+5, x, mesh), mesh, "compose")
