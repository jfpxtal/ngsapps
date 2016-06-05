from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from ngsolve import *
from ngsapps.utils import *
import scipy.constants as sciconst
import matplotlib.pyplot as plt
import random

usegeo = "1d"

order = 1
maxh = 0.2

initial_roughness = 20.0

# time step and end
tau = 1e-8
tend = 3

# # stiffness of linker proteins
# k = 1e-4 # N / m
# # linker attachment rate
# kon = 1e4 # 1 / s
# # linker detachment rate
# koff = 10 # 1 / s
# # linker bond length
# delta = 1e-9 # m
# # density of available linkers
# rho0 = 1e14 # 1 / m^-2
# # negative outward pressure ?
# f = 1e-5

# # temperature
# T = 293 # ??
# beta = 1 / (sciconst.Boltzmann * T)

# stiffness of linker proteins
k = 1 # N / m
# linker attachment rate
kon = 1 # 1 / s
# linker detachment rate
koff = 1 # 1 / s
# linker bond length
delta = 1 # m
# density of available linkers
rho0 = 1 # 1 / m^-2
# negative outward pressure ?
f = 1e-2

# temperature
T = 1 # ??
beta = 1

# diffusion coefficients
kappa = 1e-99
gamma = 1e-99

vtkoutput = False

def sqr(x):
    return x * x

# add gaussians with random positions and widths until we reach total mass >= 0.5
def set_initial_conditions(result_gridfunc, mesh):
    # c0 = GridFunction(result_gridfunc.space)
    c0 = CoefficientFunction(0.0)
    total_mass = 0.0
    # vec_storage = c0.vec.CreateVector()
    # vec_storage[:] = 0.0

    print("setting initial conditions")
    while total_mass < 0.5:
        print("\rtotal mass = {:10.6e}".format(total_mass), end="")
        center_x = 1 - 2 * random.random()
        center_y = 1 - 2 * random.random()
        if center_x ** 2  + center_y ** 2 > 1:
            continue
        thinness_x = initial_roughness * (1+random.random())
        thinness_y = initial_roughness * (1+random.random())
        c0 += exp(-(sqr(thinness_x) * sqr(x-center_x) + sqr(thinness_y) * sqr(y-center_y)))
        # vec_storage.data += c0.vec
        # c0.vec.data = vec_storage

        # cut off above 1.0
        result_gridfunc.Set(IfPos(c0-1.0,1.0,c0))
        total_mass = Integrate(result_gridfunc,mesh,VOL)

    print()

if usegeo == "circle":
    geo = SplineGeometry()
    MakeCircle(geo, (0,0), 1)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order)
elif usegeo == "square":
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
elif usegeo == "1d":
    netmesh = NetMesh()
    netmesh.dim = 1
    N = 100
    pnums = []
    for i in range(0, N + 1):
        pnums.append(netmesh.Add(MeshPoint(Pnt(i, 0, 0))))

    for i in range(0, N):
        netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))

    netmesh.Add(Element0D(pnums[0], index=1))
    netmesh.Add(Element0D(pnums[N], index=2))

    mesh = Mesh(netmesh)

V = H1(mesh, order=order)
fes = FESpace([V, V, V])
u, rho, mu = fes.TrialFunction()
tu, trho, tmu = fes.TestFunction()
# fes = FESpace([V, V])
# u, rho = fes.TrialFunction()
# tu, trho = fes.TestFunction()

a = BilinearForm(fes)

a += SymbolicBFI(kappa * grad(mu) * grad(tu))
a += SymbolicBFI(-gamma * grad(u) * grad(tu))
a += SymbolicBFI(-k * rho * u * tu)

a += SymbolicBFI(-kon * rho * trho)
a += SymbolicBFI(-koff * exp(beta * k * u * delta) * rho * trho)

b = BilinearForm(fes)
b += SymbolicBFI(u * tu)
b += SymbolicBFI(rho * trho)

c = BilinearForm(fes)
c += SymbolicBFI(-grad(u) * grad(tmu))

l = LinearForm(fes)
l += SymbolicLFI(f * tu)
l += SymbolicLFI(kon * trho)

b.Assemble()
c.Assemble()
l.Assemble()

mstar = a.mat.CreateMatrix()

s = GridFunction(fes)

# s.components[0].Set(CoefficientFunction(0.0))
s.components[0].Set(RandomCF(0.0,1.0))
# set_initial_conditions(s.components[0], mesh)
# s.components[1].Set(RandomCF(0.2,0.8))
s.components[1].Set(CoefficientFunction(rho0))
s.components[2].Set(CoefficientFunction(0.0))
# s.components[2].Set(RandomCF(0.0,1.0))

if usegeo == "1d":
    xs = [i for i in range(0, N + 1)]

    def get_vals(u):
        return [u(x) for x in xs]

    fig = plt.figure()
    ax_u = fig.add_subplot(211)
    line_u, = ax_u.plot(xs, get_vals(s.components[0]), "b", label="u")
    ax_rho = fig.add_subplot(212)
    line_rho, = ax_rho.plot(xs, get_vals(s.components[1]), "b", label="rho")
    plt.show(block=False)

rhs = s.vec.CreateVector()
sold = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

# Draw(s.components[2], mesh, "mu")
Draw(s.components[1], mesh, "rho")
Draw(s.components[0], mesh, "u")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[0],s.components[1]],names=["u","rho"],filename="bleb_",subdivision=4)
    vtk.Do()

input("Press any key...")
# implicit Euler
t = 0.0
while t < tend:
    print("\n\nt = {:10.6e}".format(t))
    if usegeo == "1d":
        line_u.set_ydata(get_vals(s.components[0]))
        ax_u.relim()
        ax_u.autoscale_view()
        line_rho.set_ydata(get_vals(s.components[1]))
        ax_rho.relim()
        ax_rho.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(0.05)
    # input("")
    # print(s.components[1].vec)
    # input("")

    sold.data = s.vec
    wnorm = 1e99

    # Newton solver
    while wnorm > 1e-9:
    # for it in range(5):
        rhs.data = b.mat * sold
        # print(rhs.data,1)
        # input("")
        rhs.data -= b.mat * s.vec
        # print(rhs.data,2)
        # input("")

        rhs.data -= c.mat * s.vec

        # print(rhs.data,3)
        # input("")
        a.Apply(s.vec,As)
        rhs.data += tau * As
        rhs.data += tau * l.vec
        # print(rhs.data,4)
        # input("")
        a.AssembleLinearization(s.vec)

        mstar.AsVector().data = b.mat.AsVector() + c.mat.AsVector() - tau * a.mat.AsVector()
        # mstar.AsVector().data = b.mat.AsVector() - tau * a.mat.AsVector()
        # print(b.mat.AsVector().data,5)
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print(c.mat.AsVector().data,5)
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print(mstar.AsVector().data,5)
        # input("")
        invmat = mstar.Inverse()
        # print(invmat)
        # input("")
        w.data = invmat * rhs
        # print(w.data)
        # input("")
        wnorm = w.Norm()
        print("|w| = {:7.3e} ".format(wnorm)) # ,end="")
        s.vec.data += w
        # input("")

    t += tau
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
