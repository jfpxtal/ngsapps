# http://www.math.umn.edu/~scheel/preprints/pf0.pdf

from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.geom2d import SplineGeometry, unit_square
from netgen.csg import Pnt
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *

order = 3

# L = 10
# N = 10
# L = 700
L = 200
N = 1000
dx = L / N

tau = 1
# tend = 4000
tend = -1

gamma = 0.1
alpha = 0.2
kappa = 0

vtkoutput = False

geo = SplineGeometry()
geo.AddRectangle((0,0), (200,1))
mesh = Mesh(geo.GenerateMesh(maxh=0.5))

# geo = SplineGeometry()
# geo.AddRectangle((0,0), (200,200))
# mesh = Mesh(geo.GenerateMesh(maxh=10))

# V = L2(mesh, order=order, flags={ 'dgjumps': True })
# fes = FESpace([V, V], flags={ 'dgjumps': True })
Vc = L2(mesh, dirichlet=[], order=order)
Ve = L2(mesh, dirichlet=[], order=order)
fes = FESpace([Vc, Ve], flags={ 'dgjumps': True })
# fes = FESpace([Vc, Ve])
c, e = fes.TrialFunction()
tc, te = fes.TestFunction()

a = BilinearForm(fes)
a += SymbolicBFI(grad(c) * grad(tc))
a += SymbolicBFI(e * (1 - e) * (e - alpha) * tc)
a += SymbolicBFI(gamma * c * tc)
a += SymbolicBFI(kappa * grad(e) * grad(te))
a += SymbolicBFI(-e * (1 - e) * (e - alpha) * te)
a += SymbolicBFI(-gamma * c * te)

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

d = BilinearForm(fes)
d += SymbolicBFI(0.5 * (grad(c) + grad(c.Other())) * n * (tc - tc.Other()), VOL, skeleton=True)
d += SymbolicBFI(0.5 * (grad(tc) + grad(tc.Other())) * n * (c - c.Other()), VOL, skeleton=True)
d += SymbolicBFI(-10 * order * (order+1) / h * (c - c.Other()) * (tc - tc.Other()), VOL, skeleton=True)

# a += SymbolicBFI(kappa * 0.5 * (grad(e) + grad(e.Other())) * n * (te - te.Other()), VOL, skeleton=True)
# a += SymbolicBFI(kappa * 0.5 * (grad(te) + grad(te.Other())) * n * (e - e.Other()), VOL, skeleton=True)
# a += SymbolicBFI(kappa * 10 * order * (order+1) / h * (e - e.Other()) * (te - te.Other()), VOL, skeleton=True)

b = BilinearForm(fes)
b += SymbolicBFI(c * tc)
b += SymbolicBFI(e * te)

b.Assemble()
d.Assemble()

mstar = b.mat.CreateMatrix()

s = GridFunction(fes)

# width = 1
# s.components[0].Set(0.1 * exp(-((x-100) * (x-100) + (y-100) * (y-100)) / width))
s.components[0].Set(IfPos(0.1 - x, 0.1, 0))
s.components[1].Set(CoefficientFunction(alpha))

rhs = s.vec.CreateVector()
sold = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[1],s.components[0]],names=["e","c"],filename="precipfem_",subdivision=3)
    vtk.Do()

xs = [i * dx for i in range(0, N + 1)]
ts = [0]

def get_vals(u):
    return [u(x) for x in xs]

# fig_sol = plt.figure()

# ax_e = fig_sol.add_subplot(211)
# line_e, = ax_e.plot(xs, get_vals(s.components[1]), "b", label="e")
# ax_e.legend()

# ax_c = fig_sol.add_subplot(212)
# line_c, = ax_c.plot(xs, get_vals(s.components[0]), "b", label="c")
# ax_c.legend()

# fig_mass = plt.figure()
# ax_mass = fig_mass.add_subplot(111)
# masses = [Integrate(s.components[0], mesh) + Integrate(s.components[1], mesh)]
# line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
# ax_mass.legend()

# plt.show(block=False)

# visfun = GridFunction(visV)
# visfun.Set(s.components[1])
# Draw(visfun, vismesh, "e")

Draw(s.components[0], mesh, "c")
Draw(s.components[1], mesh, "e")


input("Press any key...")
# implicit Euler
t = 0.0
it = 1
while tend < 0 or t < tend:
    print("\n\nt = {:10.2f}".format(t))
    # if it % 100 == 0:
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    #     # visfun.Set(s.components[1])
    #     line_e.set_ydata(get_vals(s.components[1]))
    #     line_c.set_ydata(get_vals(s.components[0]))
    #     ax_e.relim()
    #     ax_e.autoscale_view()
    #     ax_c.relim()
    #     ax_c.autoscale_view()
    #     ts.append(t)
    #     masses.append(Integrate(s.components[0], mesh) + Integrate(s.components[1], mesh))
    #     line_mass.set_xdata(ts)
    #     line_mass.set_ydata(masses)
    #     ax_mass.relim()
    #     ax_mass.autoscale_view()
    #     fig_sol.canvas.draw()
    #     fig_mass.canvas.draw()

    sold.data = s.vec
    wnorm = 1e99

    # Newton solver
    while wnorm > 1e-9:
        rhs.data = b.mat * sold
        rhs.data -= b.mat * s.vec
        rhs.data -= tau * d.mat * s.vec
        a.Apply(s.vec, As)
        rhs.data -= tau * As
        a.AssembleLinearization(s.vec)

        mstar.AsVector().data = b.mat.AsVector()
        mstar.AsVector().data += tau * a.mat.AsVector()
        mstar.AsVector().data += tau * d.mat.AsVector()
        invmat = mstar.Inverse()
        w.data = invmat * rhs
        wnorm = w.Norm()
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s.vec.data += w

    t += tau
    it += 1
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
