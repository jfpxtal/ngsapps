# http://www.math.umn.edu/~scheel/preprints/pf0.pdf

from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from ngsolve import *
import matplotlib.pyplot as plt

order = 3

# L = 10
# N = 10
L = 700
N = 1000
dx = L / N

tau = 0.05
tend = 4000

gamma = 0.1
alpha = 0.2
kappa = 0

vtkoutput = False

netmesh = NetMesh()
netmesh.dim = 1

pnums = []
for i in range(0, N + 1):
    pnums.append(netmesh.Add(MeshPoint(Pnt(i * dx, 0, 0))))

for i in range(0, N):
    netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))

netmesh.Add(Element0D(pnums[0], index=1))
netmesh.Add(Element0D(pnums[N], index=2))

mesh = Mesh(netmesh)


V = H1(mesh, order=order)
fes = FESpace([V, V])
c, e = fes.TrialFunction()
tc, te = fes.TestFunction()

a = BilinearForm(fes)
a += SymbolicBFI(grad(c) * grad(tc))
a += SymbolicBFI(e * (1 - e) * (e - alpha) * tc)
a += SymbolicBFI(gamma * c * tc)
a += SymbolicBFI(kappa * grad(e) * grad(te))
a += SymbolicBFI(-e * (1 - e) * (e - alpha) * te)
a += SymbolicBFI(-gamma * c * te)

b = BilinearForm(fes)
b += SymbolicBFI(c * tc)
b += SymbolicBFI(e * te)

b.Assemble()

mstar = b.mat.CreateMatrix()

s = GridFunction(fes)

s.components[0].Set(IfPos(20 - x, 0.1, 0))
s.components[1].Set(CoefficientFunction(alpha))

rhs = s.vec.CreateVector()
sold = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

Draw(s.components[0], mesh, "c")
Draw(s.components[1], mesh, "e")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[1],s.components[0]],names=["e","c"],filename="precipitation_",subdivision=3)
    vtk.Do()

xs = [i * dx for i in range(0, N + 1)]
ts = [0]

def get_vals(u):
    return [u(x) for x in xs]

fig_sol = plt.figure()

ax_e = fig_sol.add_subplot(211)
line_e, = ax_e.plot(xs, get_vals(s.components[1]), "b", label="e")
ax_e.legend()

ax_c = fig_sol.add_subplot(212)
line_c, = ax_c.plot(xs, get_vals(s.components[0]), "b", label="c")
ax_c.legend()

fig_mass = plt.figure()
ax_mass = fig_mass.add_subplot(111)
masses = [Integrate(s.components[0], mesh) + Integrate(s.components[1], mesh)]
line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
ax_mass.legend()

plt.show(block=False)

input("Press any key...")
# implicit Euler
t = 0.0
it = 1
while t < tend:
    print("\n\nt = {:10.6e}".format(t))
    if it % 100 == 0:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        line_e.set_ydata(get_vals(s.components[1]))
        line_c.set_ydata(get_vals(s.components[0]))
        ax_e.relim()
        ax_e.autoscale_view()
        ax_c.relim()
        ax_c.autoscale_view()
        ts.append(t)
        masses.append(Integrate(s.components[0], mesh) + Integrate(s.components[1], mesh))
        line_mass.set_xdata(ts)
        line_mass.set_ydata(masses)
        ax_mass.relim()
        ax_mass.autoscale_view()
        fig_sol.canvas.draw()
        fig_mass.canvas.draw()

    sold.data = s.vec
    wnorm = 1e99

    # Newton solver
    while wnorm > 1e-9:
        rhs.data = b.mat * sold
        rhs.data -= b.mat * s.vec
        a.Apply(s.vec,As)
        rhs.data -= tau * As
        a.AssembleLinearization(s.vec)

        mstar.AsVector().data = b.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse()
        w.data = invmat * rhs
        wnorm = w.Norm()
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s.vec.data += w

    t += tau
    it += 1
    # Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
