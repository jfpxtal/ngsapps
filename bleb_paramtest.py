from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from ngsolve import *
from ngsapps.utils import *
import scipy.constants as sciconst
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import math

usegeo = "circle"

order = 3
maxh = 0.1

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
# # cortical stress
# f = 5e4 # N / m

# temperature
T = 1 # K
beta = 1

# membrane bending rigidity
kappa = 1e-5 # J
# membrane surface tension
gamma = 1e-5 # N / m

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


a.Assemble()
b.Assemble()
c.Assemble()

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

rhs = s.vec.CreateVector()
s_init = s.vec.CreateVector()
s_old = s.vec.CreateVector()
s_init.data = s.vec

Draw(s.components[2], mesh, "mu")
Draw(s.components[1], mesh, "rho")
Draw(s.components[0], mesh, "u")

print("Testing k = {}\tkon = {}\tkoff = {}\tkappa = {}\tgamma = {}\n\n".format(k, kon, koff, kappa, gamma))

print("Phase 1: localize t_stat\n")

t = 0.0
it = 0
tau = 1e-10
stat_mass_ratio = 0.01
f = 1e-10

l = LinearForm(fes)
l += SymbolicLFI(f * tu)
l += SymbolicLFI(kon * trho)
l.Assemble()

mass_old = 0
while True:
    t += tau
    s_old.data = s.vec
    d.Assemble()

    rhs.data = b.mat * s.vec
    rhs.data += tau * l.vec

    mstar.AsVector().data = b.mat.AsVector() + c.mat.AsVector() - tau * (a.mat.AsVector() + d.mat.AsVector())
    invmat = mstar.Inverse(fes.FreeDofs())
    s.vec.data = invmat * rhs

    with TaskManager():
        mass = Integrate(s.components[0], mesh)

    mass_change = math.fabs(mass - mass_old)
    print("it = {}\tt = {}\ttau = {}\tmass = {:5.3e}\tchange = {:5.3e}\n".format(it, t, tau, mass, mass_change))
    if it > 0 and mass_change < stat_mass_ratio * mass:
        break

    mass_old = mass
    tau *= 2
    it += 1

t_stat_min = t - tau
t_stat_max = t
t_stat_tol = 1

print("\nFinished phase 1: t_stat_min = {}\tt_stat_max = {}\n".format(t_stat_min, t_stat_max))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print("Phase 2: binary search for t_stat\n")

it_bin_search = 0
while t_stat_max - t_stat_min > t_stat_tol:
    t_stat = (t_stat_min + t_stat_max) / 2
    tau = (t_stat_max - t_stat_min) / 2

    s.vec.data = s_old
    d.Assemble()

    rhs.data = b.mat * s.vec
    rhs.data += tau * l.vec

    mstar.AsVector().data = b.mat.AsVector() + c.mat.AsVector() - tau * (a.mat.AsVector() + d.mat.AsVector())
    invmat = mstar.Inverse(fes.FreeDofs())
    s.vec.data = invmat * rhs

    with TaskManager():
        mass = Integrate(s.components[0], mesh)

    mass_change = math.fabs(mass - mass_old)
    print("it_bin_search = {}\tt_stat_min = {}\tt_stat_max = {}\tt_stat = {}\tmass = {:5.3e}\tchange = {:5.3e}".format(it_bin_search, t_stat_min, t_stat_max, t_stat, mass, mass_change))
    if mass_change < stat_mass_ratio * mass:
        print("t_stat < t\n")
        t_stat_max = t_stat
    else:
        print("t < t_stat\n")
        t_stat_min = t_stat
        s_old.data = s.vec
        mass_old = mass

    it_bin_search += 1

# t_stat = t
tau = t_stat / 2
mass_stat = [mass]

print("\nFinished phase 2: t_stat = {}\ttau = {:5.3e}\tmass_stat = {:5.3e}\n".format(t_stat, tau, mass))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print("Phase 3: binary search for critical pressure\n")

e_min = -10
e_max = 10
m_min = 0
m_max = 10
f_min = 1e-10
f_max = 1e10
f_tol = 1e-3
additional_steps = 4

it_bin_search = 0
stable = True
while not stable or f_max - f_min > f_tol:
# while not stable or e_max > e_min or m_max - m_min > 1:
    f = (f_max + f_min) / 2
    print("it_bin_search = {}\tf_min = {:5.3e}\tf_max = {:5.3e}\tf = {:5.3e}\n".format(it_bin_search, f_min, f_max, f))
    # exponent = math.floor((e_max + e_min) / 2)
    # mant = (m_max + m_min) / 2
    # f = mant * 10 ** exponent
    # f_min = m_min * 10 ** e_min
    # f_max = m_max * 10 ** e_max
    # print("it_bin_search = {}\te_min = {}\te_max = {}\tm_min = {}\tm_max = {}\tf = {:5.3e}\n".format(it_bin_search, e_min, e_max, m_min, m_max, f))

    l = LinearForm(fes)
    l += SymbolicLFI(f * tu)
    l += SymbolicLFI(kon * trho)
    l.Assemble()

    s.vec.data = s_init
    t = 0
    it = 0
    mass_old = 0
    stable = True
    while t <= t_stat + additional_steps * tau:
        d.Assemble()

        rhs.data = b.mat * s.vec
        rhs.data += tau * l.vec

        mstar.AsVector().data = b.mat.AsVector() + c.mat.AsVector() - tau * (a.mat.AsVector() + d.mat.AsVector())
        invmat = mstar.Inverse(fes.FreeDofs())
        s.vec.data = invmat * rhs

        with TaskManager():
            mass = Integrate(s.components[0], mesh)

        if math.isnan(mass):
            # instability, pressure too high
            print("Instability f = {:5.3e}\tt = {}\n".format(f, t))
            stable = False
            break

        mass_change = math.fabs(mass - mass_old)
        print("it = {}\tt = {}\tf = {:5.3e}\tmass = {:5.3e}\tchange = {:5.3e}\n".format(it, t, f, mass, mass_change))
        if it > 0 and mass_change < stat_mass_ratio * mass:
            print("Stationary point reached: f = {:5.3e}\tmass = {:5.3e}\tt = {}".format(f, mass, t))
            break

        mass_old = mass
        t += tau
        it += 1

    mass_stat.append(mass)
    if not stable or t > t_stat + additional_steps * tau:
        print("f_crit < f")
        # if e_max == e_min:
        #     m_max = mant
        # e_max = exponent
        f_max = f
    else:
        print("f < f_crit")
        # if e_max == e_min:
        #     m_min = mant
        # e_min = exponent
        f_min = f

    it_bin_search += 1

print("\n\nFinished: f_crit = {:5.3e}\tf_min = {:5.3e}\tf_max = {:5.3e}".format(f, f_min, f_max))
