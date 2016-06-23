# solve the Poisson equation -Delta u = f
# with Dirichlet boundary condition u = 0

from ngsolve import *
from netgen.geom2d import unit_square
from netgen.csg import unit_cube
from ngsapps.utils import Lagrange

ngsglobals.msg_level = 1

dim = 2
# generate a triangular mesh of mesh-size 0.2
if dim == 2:
    mesh = Mesh (unit_square.GenerateMesh(maxh=0.2))
else:
    mesh = Mesh (unit_cube.GenerateMesh(maxh=0.2))

# H1-conforming finite element space
V = Lagrange(mesh, order=3, dirichlet=[1,2,3,4,5,6])

# the right hand side
f = LinearForm (V)
if dim == 2:
    f += Source (32 * (y*(1-y)+x*(1-x)))
else:
    f += Source (32 * (x*(1-x)*y*(1-y)+x*(1-x)*z*(1-z)+y*(1-y)*z*(1-z)))

# the bilinear-form 
a = BilinearForm (V, symmetric=True)
a += Laplace (1)


a.Assemble()
f.Assemble()

# the solution field 
u = GridFunction (V)
u.Set(1*x)

f.vec.data -= a.mat * u.vec
u.vec.data += a.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky") * f.vec
# print (u.vec)


# plot the solution (netgen-gui only)
Draw (u)
Draw (-u.Deriv(), mesh, "Flux")

if dim == 2:
    exact = 1*x + 16*x*(1-x)*y*(1-y)
else:
    exact = 1*x + 16*x*(1-x)*y*(1-y)*z*(1-z)
print ("L2-error:", sqrt (Integrate ( (u-exact)*(u-exact), mesh)))

