from ngsolve import *
from netgen.geom2d import SplineGeometry, unit_square
from ngsapps.utils import *
import matplotlib.pyplot as plt
import numpy as np

order = 3
maxh = 1

vtkoutput = False

# time step and end
tau = 50
tend = -1

# diffusion coefficient for rho
DT = 0.001

# determines how quickly the average local swim speed
# decreases with the density
alpha = 0.38

# positive parameters which ensure that, at steady state
# and in the absence of any density or velocity gradients,
# W vanishes everywhere
gamma1 = 0.04
gamma2 = 0

# effective diffusion coefficent for W
# ensures continuity of this field
k = 0.06

# self-advection
w1 = 0
# active pressure, >0
w2 = 10

netmesh = NetMesh()
netmesh.dim = 1
L = 600
N = int(600/maxh)
pnums = []
for i in range(0, N + 1):
    pnums.append(netmesh.Add(MeshPoint(Pnt(L * i / N, 0, 0))))

for i in range(0, N):
    netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))

netmesh.Add(Element0D(pnums[0], index=1))
netmesh.Add(Element0D(pnums[N], index=2))
mesh = Mesh(netmesh)

v0 = 0.2
vmin = 0.001
v = IfPos(100-x, v0,
            IfPos(200-x, vmin+(x/100-1)*(v0-vmin),
                IfPos(400-x, v0,
                        IfPos(500-x, v0+(x/100-4)*(vmin-v0), v0))))

vdx = IfPos(100-x, 0,
            IfPos(200-x, (v0-vmin)/100,
                    IfPos(400-x, 0,
                        IfPos(500-x, (vmin-v0)/100, 0))))

fesRho = H1(mesh, order=order)
fesW = H1(mesh, order=order-1)
fes = FESpace([fesRho, fesW])

rho, W = fes.TrialFunction()
trho, tW = fes.TestFunction()

g = GridFunction(fes)
grho, gW = g.components
vbar = v * exp(-alpha*grho)
gradvbar = vdx*exp(-alpha*grho) - alpha*grad(grho)*vbar
WdotdelW = gW*grad(W)
gradnormWsq = 2*gW*grad(W)

# initial values
# grho.Set(exp(-sqr(x)-sqr(y)))
# measure = Integrate(CoefficientFunction(1), mesh)
# grho.Set(CoefficientFunction(1/measure))
grho.Set(CoefficientFunction(1))
gW.Set(CoefficientFunction(0))

a = BilinearForm(fes)

# equation for rho
# TODO: boundary terms from partial integration?
# TODO: separate terms which need to be reassembled at every time step
a += SymbolicBFI(vbar*W*grad(trho) - DT*grad(rho)*grad(trho))
# a += SymbolicBFI(-gradvbar*W*trho - vbar*grad(W)*trho - DT*grad(rho)*grad(trho))

# equation for W
# a += SymbolicBFI(0.5*vbar*rho*grad(tW) - gamma1*W*tW
#                  -gamma2*sqr(gW)*W*tW - k*grad(W)*grad(tW)
#                  -w1*WdotdelW*tW + w2*gradnormWsq*tW)
a += SymbolicBFI(-0.5*(gradvbar*rho + vbar*grad(rho))*tW - gamma1*W*tW
                 -gamma2*sqr(gW)*W*tW - k*grad(W)*grad(W)
                 -w1*WdotdelW*tW + w2*gradnormWsq*tW)

m = BilinearForm(fes)
m += SymbolicBFI(rho*trho + W*tW)

m.Assemble()

rhs = g.vec.CreateVector()
mstar = m.mat.CreateMatrix()

xs = np.linspace(0, L, L/maxh)
mips = [mesh(x) for x in xs]
plt.plot(xs, [v(i)/v0 for i in mips])
line, = plt.plot(xs, [grho(i) for i in mips])
plt.show(block=False)

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh, coefs=[g.components[0], g.components[1]],names=["rho", "W"], filename="instab/instab",subdivision=3)
    vtk.Do()

input("Press any key...")
t = 0.0
k = 0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
        print("\nt = {:10.6e}".format(t))
        t += tau
        print('Assembling a...')
        a.Assemble()
        print('...done')

        rhs.data = m.mat * g.vec
        mstar.AsVector().data = m.mat.AsVector() - tau*a.mat.AsVector()
        invmat = mstar.Inverse(fes.FreeDofs())
        g.vec.data = invmat * rhs

        if k % 20 == 0:
            line.set_ydata([grho(i) for i in mips])
            # plt.gcf().canvas.draw()
            plt.pause(0.05)

        if vtkoutput:
            vtk.Do()

        k += 1
