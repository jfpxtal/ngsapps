from ngsolve import *
from netgen.geom2d import SplineGeometry, unit_square
from ngsapps.utils import *
import matplotlib.pyplot as plt
import numpy as np
from ngsapps.plotting import *
from ngsapps.limiter import *

order = 1
maxh = 1

vtkoutput = False

# time step and end
tau = 50
# tau = 50000
tend = -1

# diffusion coefficient for rho
DT = 0.001

# determines how quickly the average local swim speed
# decreases with the density
alpha = 0.38

# positive parameters which ensure that, at steady state
# and in the absence of any density or velocity gradients,
# W vanishes everywhere
# gamma1 = 0.06
gamma1 = 0.04
gamma2 = 0

# effective diffusion coefficent for W
# ensures continuity of this field
# k = 0.06
k = 2

# self-advection
w1 = 0
# active pressure, >0
w2 = 10
# w2 = 0

mesh = Mesh(Make1DMesh(0, 600, maxh, True))
# mesh = Mesh(Make1DMesh(-50, 300, 0.2, True))

v0 = 0.2
vmin = 0.001

# # full profile, discontinuous
# v = IfPos(100-x, v0,
#             IfPos(200-x, vmin+(x/100-1)*(v0-vmin),
#                 IfPos(400-x, v0,
#                         IfPos(500-x, v0+(x/100-4)*(vmin-v0), v0))))

# vdx = IfPos(100-x, 0,
#             IfPos(200-x, (v0-vmin)/100,
#                     IfPos(400-x, 0,
#                         IfPos(500-x, (vmin-v0)/100, 0))))

# full profile, continuous
smear = 20
v = IfPos(100-smear-x, v0,
          IfPos(100-x, v0+(x-100+smear)/smear*(vmin-v0),
            IfPos(200-x, vmin+(x/100-1)*(v0-vmin),
                IfPos(400-x, v0,
                        IfPos(500-x, v0+(x/100-4)*(vmin-v0),
                              IfPos(500+smear-x, vmin+(x-500)/smear*(v0-vmin), v0))))))

vdx = IfPos(100-smear-x, 0,
          IfPos(100-x, (vmin-v0)/smear,
            IfPos(200-x, (v0-vmin)/100,
                IfPos(400-x, 0,
                        IfPos(500-x, (vmin-v0)/100,
                              IfPos(500+smear-x, (v0-vmin)/smear, 0))))))

# single sawtooth, discontinuous
# v = IfPos(100-x, v0,
#             IfPos(200-x, vmin+(x-100)/100*(v0-vmin), v0))

# vdx = IfPos(100-x, 0,
#             IfPos(200-x, (v0-vmin)/100, 0))

# # single sawtooth, continuous
# smear = 10
# v = IfPos(100-smear-x, v0,
#             IfPos(100-x, v0+(x-100+smear)/smear*(vmin-v0),
#                 IfPos(200-x, vmin+(x-100)/100*(v0-vmin), v0)))

# vdx = IfPos(100-smear-x, 0,
#             IfPos(100-x, (vmin-v0)/smear,
#                 IfPos(200-x, (v0-vmin)/100, 0)))

fesRho = Periodic(H1(mesh, order=order))
fesW = Periodic(H1(mesh, order=order-1))
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

n = specialcf.normal(mesh.dim)

a = BilinearForm(fes)

# equation for rho
# TODO: boundary terms from partial integration?
# TODO: separate terms which need to be reassembled at every time step
a += SymbolicBFI(vbar*W*grad(trho) - DT*grad(rho)*grad(trho))
# a += SymbolicBFI(-vbar*W*n*trho + DT*grad(rho)*n*trho, BND)
# a += SymbolicBFI(-gradvbar*W*trho - vbar*grad(W)*trho - DT*grad(rho)*grad(trho))

# equation for W
a += SymbolicBFI(0.5*vbar*rho*grad(tW) - gamma1*W*tW
                 -gamma2*sqr(gW)*W*tW - k*grad(W)*grad(tW)
                 -w1*WdotdelW*tW + w2*gradnormWsq*tW)
# a += SymbolicBFI(-0.5*(gradvbar*rho + vbar*grad(rho))*tW - gamma1*W*tW
#                  -gamma2*sqr(gW)*W*tW - k*grad(W)*grad(tW)
#                  -w1*WdotdelW*tW + w2*gradnormWsq*tW)

m = BilinearForm(fes)
m += SymbolicBFI(rho*trho + W*tW)

m.Assemble()

rhs = g.vec.CreateVector()
mstar = m.mat.CreateMatrix()

mplmesh = MPLMesh1D(mesh)
mplmesh.Plot(v/v0)
lineRho = mplmesh.Plot(grho)
plt.figure()
lineW = mplmesh.Plot(gW/grho)
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

        # stabilityLimiter(grho, p1fes)        nope, need L2 DG
        # stabilityLimiter(gW, p1fes)
        # nonnegativityLimiter(grho, p1fes)

        if k % 20 == 0:
            lineRho.Redraw()
            lineW.Redraw()
            plt.gca().relim()
            plt.gca().autoscale_view()
            # plt.gcf().canvas.draw()
            plt.pause(0.05)

        if vtkoutput:
            vtk.Do()

        k += 1
