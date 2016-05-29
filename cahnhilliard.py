from netgen.geom2d import unit_square
from ngsolve import *
import random

order = 3

initial_roughness = 20.0

tau = 1e-5
tend = 3

lamdba = 1e-2
M = 1

vtkoutput = False

def sqr(x):
    return x * x

# add gaussians with random positions and widths until we reach total mass >= 0.5
def set_initial_conditions(result_gridfunc):
    c0 = GridFunction(result_gridfunc.space)
    total_mass = 0.0
    vec_storage = c0.vec.CreateVector()
    vec_storage[:] = 0.0

    print("setting initial conditions")
    while total_mass < 0.5:
        print("\rtotal mass = {:10.6e}".format(total_mass), end="")
        center_x = random.random()
        center_y = random.random()
        thinness_x = initial_roughness * (1+random.random())
        thinness_y = initial_roughness * (1+random.random())
        c0.Set(exp(-(sqr(thinness_x) * sqr(x-center_x) + sqr(thinness_y) * sqr(y-center_y))))
        vec_storage.data += c0.vec
        c0.vec.data = vec_storage

        # cut off above 1.0
        result_gridfunc.Set(IfPos(c0-1.0,1.0,c0))
        total_mass = Integrate(s.components[0],mesh,VOL)

    print()


mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

V = H1(mesh, order=order)
fes = FESpace([V, V])
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

set_initial_conditions(s.components[0])
s.components[1].Set(CoefficientFunction(0.0))

rhs = s.vec.CreateVector()
sold = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

Draw(s.components[1], mesh, "mu")
Draw(s.components[0], mesh, "c")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[1],s.components[0]],names=["mu","c"],filename="cahnhilliard_",subdivision=3)
    vtk.Do()

input("Press any key...")
# implicit Euler
t = 0.0
while t < tend:
    print("\n\nt = {:10.6e}".format(t))

    sold.data = s.vec
    wnorm = 1e99

    # newton solver
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
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s.vec.data += w

    t += tau
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
