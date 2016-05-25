from netgen.geom2d import unit_square
from ngsolve import *
import random

order = 3

tau = 1e-5
tend = 3

lamdba = 1e-2
M = 1

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

V1 = H1(mesh, order=order) # , dirichlet=[1,2,3,4])
V2 = H1(mesh, order=order) #, dirichlet=[1,2,3,4])

fes = FESpace([V1, V2])
c, mu = fes.TrialFunction()
q, v = fes.TestFunction()

a = BilinearForm(fes)
a += SymbolicBFI(tau * M * grad(mu) * grad(q))
a += SymbolicBFI(mu * v)
a += SymbolicBFI(-200 * (c - 3 * c * c + 2 * c * c * c) * v)
a += SymbolicBFI(-lamdba * grad(c) * grad(v))

b = BilinearForm(fes)
b += SymbolicBFI(c * q)

b.Assemble()

mstar = b.mat.CreateMatrix()

s = GridFunction(fes)

# initial conditions
### TODO
# s.components[0].Set(sin(3*x)*cos(4*y))
for i in range(len(s.components[0].vec)):
    s.components[0].vec[i]=0.63 + 0.02 * (0.5 - random.random())
# s.components[0].Set(0.5+0.5*sin(4e6*(x*x-0.5*y))*sin(5e6*(y*y-0.5*x)))
# s.components[0].Set(x - x + 0.63 + 0.02 * (0.5 - random.random()))
s.components[1].Set(CoefficientFunction(0.0))

rhs = s.vec.CreateVector()
sold = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

Draw(s.components[1], mesh, "mu")
Draw(s.components[0], mesh, "c")

input("")
# implicit Euler
t = 0.0
while t < tend:
    print("t = {:10.6e}".format(t))#,end="")

    sold.data = s.vec
    wnorm = 1e99

    while wnorm > 1e-9:
        rhs.data = b.mat * sold
        rhs.data -= b.mat * s.vec
        a.Apply(s.vec,As)
        rhs.data -= As
        a.AssembleLinearization(s.vec)

        mstar.AsVector().data = b.mat.AsVector() + a.mat.AsVector()
        invmat = mstar.Inverse()
        w.data = invmat * rhs
        wnorm = w.Norm()
        print("|w| = {:7.3e}".format(wnorm),end="")
        s.vec.data += w

    t += tau
    Redraw(blocking=True)
