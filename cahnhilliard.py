from netgen.geom2d import unit_square
from ngsolve import *
import random

order = 3

tau = 5e-6
tend = 3

lamdba = 1e-2
M = 1

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

V1 = H1(mesh, order=order) # , dirichlet=[1,2,3,4])
V2 = H1(mesh, order=order) #, dirichlet=[1,2,3,4])

fes = FESpace([V1, V2])
c, mu = fes.TrialFunction()
q, v = fes.TestFunction()
print("b")
a = BilinearForm(fes)
a += SymbolicBFI(tau * M * grad(mu) * grad(q))
a += SymbolicBFI(mu * v)
a += SymbolicBFI(-200 * (c - 2 * c * c * c) * v)
a += SymbolicBFI(-lamdba * grad(c) * grad(v))

b = BilinearForm(fes)
b += SymbolicBFI(c * q)
print("c")
b.Assemble()
print("f")
mstar = b.mat.CreateMatrix()

s = GridFunction(fes)
print("d")
# initial conditions
s.components[0].Set(x - x + 0.63 + 0.02 * (0.5 - random.random()))
print("e")
s.components[1].Set(x - x)

rhs = s.vec.CreateVector()

Draw(s.components[1], mesh, "mu")
Draw(s.components[0], mesh, "c")

# implicit Euler
print("a")
t = 0.0
while t < tend:
    print("\r t = {:10.6e}".format(t),end="")
    input("")

    rhs.data = b.mat * s.vec
    a.AssembleLinearization(rhs)

    mstar.AsVector().data = b.mat.AsVector() + a.mat.AsVector()
    invmat = mstar.Inverse()

    s.vec.data = invmat * rhs

    t += tau
    Redraw(blocking=True)

# for it in range(5):
#     print ("Iterateion",it)
#     a.Apply(u.vec, r)
#     a.AssembleLinearization(u.vec)

#     w.data = a.mat.Inverse(V.FreeDofs()) * r.data
#     print ("|w| =", w.Norm())
#     u.vec.data -= w

#     Draw(u)
#     input("<press a key>")
