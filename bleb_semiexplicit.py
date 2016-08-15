from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from ngsolve import *
from ngsapps.utils import *
import scipy.constants as sciconst
import matplotlib.pyplot as plt
import random
import multiprocessing as mp

usegeo = "circle"

order = 3
maxh = 0.1

initial_roughness = 20.0

# time step and end
tau = 1e0
tend = -1

# # stiffness of linker proteins
# k = 1e-4 # N / m
# # linker attachment rate
# kon = 1e4 # 1 / s
# # linker detachment rate
# koff = 10 # 1 / s
# # linker bond length
# delta = 1e-9 # m
# # density of available linkers
# rho0 = 1e14 # 1 / m^2
# # cortical stress
# f = 2e-4 # N / m

# # temperature
# T = 293 # K
# beta = 1 / (sciconst.Boltzmann * T)

# # membrane bending rigidity
# kappa = 1e-19 # J
# # membrane surface tension
# gamma = 5e-5 # N / m

# stiffness of linker proteins
k = 1 # N / m
# linker attachment rate
kon = 1 # 1 / s
# linker detachment rate
koff = 1 # 1 / s
# linker bond length
delta = 1 # m
# density of available linkers
rho0 = 0 # 1 / m^2
# cortical stress
f = 2e-1 # N / m

# temperature
T = 1 # K
beta = 1

# membrane bending rigidity
kappa = 1e-5 # J
# membrane surface tension
gamma = 1e-5 # N / m

vtkoutput = False

def sqr(x):
    return x * x

# add gaussians with random positions and widths until we reach total mass >= 0.5
def set_initial_conditions(result_gridfunc, mesh):
    c0 = CoefficientFunction(0.0)
    total_mass = 0.0

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
    N = 1000
    pnums = []
    for i in range(0, N + 1):
        pnums.append(netmesh.Add(MeshPoint(Pnt(i * 1 / N, 0, 0))))

    for i in range(0, N):
        netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))

    netmesh.Add(Element0D(pnums[0], index=1))
    netmesh.Add(Element0D(pnums[N], index=2))

    mesh = Mesh(netmesh)

UV = H1(mesh, order=order, dirichlet=[1,2,3,4])
V = H1(mesh, order=order)
fes = FESpace([UV, V, V])
u, rho, mu = fes.TrialFunction()
tu, trho, tmu = fes.TestFunction()
# fes = FESpace([V, V])
# u, rho = fes.TrialFunction()
# tu, trho = fes.TestFunction()

a = BilinearForm(fes)

a += SymbolicBFI(kappa * grad(mu) * grad(tu))
a += SymbolicBFI(-gamma * grad(u) * grad(tu))

a += SymbolicBFI(-kon * rho * trho)

b = BilinearForm(fes)
b += SymbolicBFI(u * tu)
b += SymbolicBFI(rho * trho)

c = BilinearForm(fes)
c += SymbolicBFI(grad(u) * grad(tmu))
c += SymbolicBFI(mu * tmu)

l = LinearForm(fes)
l += SymbolicLFI(f * tu)
l += SymbolicLFI(kon * trho)

a.Assemble()
b.Assemble()
c.Assemble()
l.Assemble()

mstar = b.mat.CreateMatrix()

s = GridFunction(fes)

d = BilinearForm(fes)
d += SymbolicBFI(-k * s.components[1] * u * tu)
d += SymbolicBFI(-koff * exp(beta * k * s.components[0] * delta) * rho * trho)

s.components[0].Set(CoefficientFunction(0.0))
# s.components[0].Set(RandomCF(0.0,1.0))
# set_initial_conditions(s.components[0], mesh)
# s.components[1].Set(RandomCF(0.2,0.8))
# s.components[1].Set(CoefficientFunction(rho0))
s.components[1].Set(0.5 * (cos(10 * (x + y)) + 1))
# s.components[1].Set(IfPos(x,1.0,0.0))
s.components[2].Set(CoefficientFunction(0.0))
# s.components[2].Set(RandomCF(0.0,1.0))

if usegeo == "1d":
    xs = [i * 1 / N for i in range(0, N + 1)]

    def get_vals(u):
        return [u(x) for x in xs]

    fig = plt.figure()
    ax_u = fig.add_subplot(211)
    line_u, = ax_u.plot(xs, get_vals(s.components[0]), "b", label="u")
    ax_rho = fig.add_subplot(212)
    line_rho, = ax_rho.plot(xs, get_vals(s.components[1]), "b", label="rho")
    plt.show(block=False)

rhs = s.vec.CreateVector()

Draw(s.components[2], mesh, "mu")
Draw(s.components[1], mesh, "rho")
Draw(s.components[0], mesh, "u")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[0],s.components[1]],names=["u","rho"],filename="bleb_semiexplicit_",subdivision=3)
    vtk.Do()

def plot_proc(massinit, t_sh, mass_sh):
    import matplotlib.pyplot as plt
    ts = [0]
    masses = [massinit]
    fig_mass = plt.figure()
    ax_mass = fig_mass.add_subplot(111)
    line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;u$")
    ax_mass.legend(loc='upper left')

    plt.show(block=False)
    while True:
        with t_sh.get_lock(), mass_sh.get_lock():
            t = t_sh.value
            mass = mass_sh.value
        if t == -1:
            break
        elif t != ts[-1]:
            ts.append(t)
            masses.append(mass)
            line_mass.set_xdata(ts)
            line_mass.set_ydata(masses)
            ax_mass.relim()
            ax_mass.autoscale_view()

        plt.pause(0.05)

    plt.show()

t_sh = mp.Value('d', 0.0)
mass_sh = mp.Value('d', 0)
proc = mp.Process(target=plot_proc, args=(Integrate(s.components[0], mesh), t_sh, mass_sh))

proc.start()
input("Press any key...")
# implicit Euler
t = 0.0
it = 0
while tend < 0 or t < tend - dt / 2:
    print("\n\nt = {:10.6e}".format(t))
    if usegeo == "1d" and it % 50 == 0:
        line_u.set_ydata(get_vals(s.components[0]))
        ax_u.relim()
        ax_u.autoscale_view()
        line_rho.set_ydata(get_vals(s.components[1]))
        ax_rho.relim()
        ax_rho.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(0.05)

    d.Assemble()

    rhs.data = b.mat * s.vec
    rhs.data += tau * l.vec

    mstar.AsVector().data = b.mat.AsVector() + c.mat.AsVector() - tau * (a.mat.AsVector() + d.mat.AsVector())
    # mstar.AsVector().data = b.mat.AsVector() - tau * (a.mat.AsVector() + d.mat.AsVector())
    invmat = mstar.Inverse(fes.FreeDofs())
    s.vec.data = invmat * rhs

    t += tau
    it += 1
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
    with TaskManager():
        mass = Integrate(s.components[0], mesh)
    with t_sh.get_lock(), mass_sh.get_lock():
        t_sh.value = t
        mass_sh.value = mass
