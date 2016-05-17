# coupled volume-surface reaction-diffusion process
# http://arxiv.org/abs/1511.00846

from ngsolve import *
from netgen.geom2d import MakeCircle
from netgen.geom2d import SplineGeometry

ngsglobals.msg_level = 1

order = 3

# time step and end
tau = 1e-2
tend = 3.0

# model parameters
# diffusion rates
dL = 0.01
dP = 0.02
# surface diffusion rates
dl = 0.02
dp = 0.04
# reaction rates
alpha = 1
beta = 2
gamma = 2
lamdba = 4
sigma = 3
xi = 1
kappa = 2.5
eta = (alpha * xi * sigma * lamdba) / (beta * gamma * kappa)

def MyCircle (geo, c, r, **args):
    cx,cy = c
    pts = [geo.AppendPoint(*p) for p in [(cx,cy-r), (cx+r,cy-r), (cx+r,cy), (cx+r,cy+r), \
                                         (cx,cy+r), (cx-r,cy+r), (cx-r,cy), (cx-r,cy-r)]]
    for p1,p2,p3 in [(0,1,2), (2,3,4), (4, 5, 6)]:
        geo.Append( ["spline3", pts[p1], pts[p2], pts[p3]], bc=1)
    geo.Append( ["spline3", pts[6], pts[7], pts[0]], bc=2)

# geometry and mesh
circ = SplineGeometry()
MyCircle(circ, (0,0), 1)
mesh = Mesh(circ.GenerateMesh(maxh=0.1))
mesh.Curve(order)

# H1-conforming finite element spaces
# inside the bulk:
VL = H1(mesh, order=order)
VP = H1(mesh, order=order)
VExt = H1(mesh, order=order, dirichlet=[1,2])
# on the surface:
Vl = H1(mesh, order=order, flags={"definedon": [], "definedonbound": [1, 2], "dirichlet": [1, 2]})
Vp = H1(mesh, order=order, flags={"definedon": [], "definedonbound": [2], "dirichlet": [2]})

# construct compound finite element space
fes = FESpace([VL, VP, Vl, Vp])

# get trial and test functions...
L,P,l,p = fes.TrialFunction()
tL,tP,tl,tp = fes.TestFunction()

# ...and their derivatives
# inside the bulk:
gradL = L.Deriv()
gradP = P.Deriv()
gradtL = tL.Deriv()
gradtP = tP.Deriv()

# on the surface:
gradl = l.Trace().Deriv()
gradp = p.Trace().Deriv()
gradtl = tl.Trace().Deriv()
gradtp = tp.Trace().Deriv()

# first bilinear form
a = BilinearForm (fes, symmetric=True)
a += SymbolicBFI(dL * gradL * gradtL)
a += SymbolicBFI(dP * gradP * gradtP)
a += SymbolicBFI(dl * gradl * gradtl, BND)
a += SymbolicBFI(dp * gradp * gradtp, BND, definedon=[1])
a += SymbolicBFI((beta * L - alpha * P) * (tL - tP))
a += SymbolicBFI((lamdba * L - gamma * l) * (tL - tl), BND)
a += SymbolicBFI((eta * P - xi * p) * (tP - tp), BND, definedon=[1])
a += SymbolicBFI((sigma * l - kappa * p) * (tl - tp), BND, definedon=[1])

# second bilinear form
c = BilinearForm(fes, symmetric=True)
c += SymbolicBFI(L * tL)
c += SymbolicBFI(P * tP)
c += SymbolicBFI(l * tl, BND)
c += SymbolicBFI(p * tp, BND, definedon=[1])

a.Assemble()
c.Assemble()

# the solution field
s = GridFunction(fes)

# initial conditions
# bulk components:
s.components[0].Set(x * sin(x + 1) + 0.5)
s.components[1].Set((2 - x) * cos(x + 1) + 0.5)
# surface components:
s.components[2].Set(0.3 * (2 - y) + 1, boundary=True)
s.components[3].Set(0.4 * y + 1, boundary=True)

# build matrix for implicit Euler
mstar = a.mat.CreateMatrix()
mstar.AsVector().data = c.mat.AsVector() + tau * a.mat.AsVector()

invmat = mstar.Inverse()
rhs = s.vec.CreateVector()

ext_p = GridFunction(VExt)
ext_l = GridFunction(VExt)

ext_l.Set(s.components[2],boundary=True)
ext_p.Set(s.components[3],boundary=True)

Draw(ext_p, mesh, "ext_p")
Draw(ext_l, mesh, "ext_l")

Draw(s.components[1], mesh, "P")
Draw(s.components[0], mesh, "L")

# implicit Euler
t = 0.0
while t < tend:
    print("t=", t)
    input("")

    rhs.data = c.mat * s.vec
    s.vec.data = invmat * rhs

    ext_l.Set(s.components[2],boundary=True)
    ext_p.Set(s.components[3],boundary=True)
    t += tau
    Redraw(blocking=True)
