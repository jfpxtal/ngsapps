# coupled volume-surface reaction-diffusion process
# http://arxiv.org/abs/1511.00846

from ngsolve import *
from netgen.geom2d import MakeCircle
from netgen.geom2d import SplineGeometry

ngsglobals.msg_level = 1

order = 3

# time step and end
tau = 1e-2
tend = 1.0

# model parameters
dL = 1
dP = 1
dl = 1
dp = 1
alpha = 1
beta = 1
gamma = 1
eta = 1
kappa = 1
lamdba = 1
xi = 1
sigma = 1

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
mesh = Mesh(circ.GenerateMesh(maxh=0.2))
mesh.Curve(order)

# H1-conforming finite element spaces
# inside the bulk:
VL = H1(mesh, order=order)
VP = H1(mesh, order=order)
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
a += SymbolicBFI(dp * gradp * gradtp, BND, definedon=[2]) # !!!!!!!!!!!!!!!!!!!!
a += SymbolicBFI((beta * L - alpha * P) * (tL - tP))
a += SymbolicBFI((lamdba * L - gamma * l) * (tL - tl), BND)
a += SymbolicBFI((eta * P - xi * p) * (tP - tp), BND, definedon=[2]) # !!!!!!
a += SymbolicBFI((sigma * l - kappa * p) * (tl - tp), BND, definedon=[2]) # !!!!!!!!

# second bilinear form
c = BilinearForm(fes, symmetric=True)
c += SymbolicBFI(L * tL)
c += SymbolicBFI(P * tP)
c += SymbolicBFI(l * tl, BND)
c += SymbolicBFI(p * tp, BND, definedon=[2]) # !!!!!!!!!!

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

alldofs = BitArray(fes.ndof)
for i in range(fes.ndof):
    alldofs.Set(i)

invmat = mstar.Inverse(alldofs)
rhs = s.vec.CreateVector()

Draw(s.components[3], mesh, "p")
Draw(s.components[2], mesh, "l")
Draw(s.components[1], mesh, "P")
Draw(s.components[0], mesh, "L")

# implicit Euler
t = 0.0
while t < tend:
    print("t=", t)
    input("")

    rhs.data = c.mat * s.vec
    s.vec.data = invmat * rhs

    t += tau
    Redraw(blocking=True)
