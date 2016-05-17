# coupled volume-surface reaction-diffusion process
# http://arxiv.org/abs/1511.00846

from ngsolve import *
from netgen.geom2d import MakeCircle
from netgen.geom2d import SplineGeometry

ngsglobals.msg_level = 1

order = 3

# time step and end
tau = 1e-3
tend = 1.0

# model parameters
dL = 0.01
dl = 0.02
gamma = 2.0
lamdba = 4.0

# geometry and mesh
circ = SplineGeometry()
MakeCircle(circ, (0,0), 1, bc=1)
mesh = Mesh(circ.GenerateMesh(maxh=0.2))
mesh.Curve(order)

# H1-conforming finite element spaces
# inside the bulk:
Vbulk = H1(mesh, order=order)
# on the surface:
Vsurface = H1(mesh, order=order, flags={"definedon": [], "definedonbound": [1], "dirichlet": [1]})

# construct compound finite element space
fes = FESpace([Vbulk, Vsurface])

# get trial and test functions...
L,l = fes.TrialFunction()
v,w = fes.TestFunction()

# ...and their derivatives
# inside the bulk:
gradL = L.Deriv()
gradv = v.Deriv()

# on the surface:
gradl = l.Trace().Deriv()
gradw = w.Trace().Deriv()

# first bilinear form
a = BilinearForm (fes, symmetric=True)
a += SymbolicBFI(dL * gradL * gradv)
# boundary terms
a += SymbolicBFI(dl * gradl * gradw, BND)
a += SymbolicBFI((lamdba * L - gamma * l) * (v - w), BND)

# second bilinear form
c = BilinearForm(fes, symmetric=True)
c += SymbolicBFI(L * v)
# boundary term
c += SymbolicBFI(l * w, BND)

a.Assemble()
c.Assemble()

# the solution field
s = GridFunction(fes)

# initial conditions
# bulk component:
s.components[0].Set(0.5 * (x * x + y * y))
# surface component:
s.components[1].Set(0.5 * (1 + x), boundary=True)

# build matrix for implicit Euler
mstar = a.mat.CreateMatrix()
mstar.AsVector().data = c.mat.AsVector() + tau * a.mat.AsVector()

invmat = mstar.Inverse()
rhs = s.vec.CreateVector()

Draw(s.components[1], mesh, "l")
Draw(s.components[0], mesh, "L")

# implicit Euler
t = 0.0
while t < tend:
    print("\r t = {:10.6e}".format(t),end="")

    rhs.data = c.mat * s.vec
    s.vec.data = invmat * rhs

    t += tau
    Redraw(blocking=True)
