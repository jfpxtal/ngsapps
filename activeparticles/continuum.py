from ngsolve import *
from netgen.geom2d import SplineGeometry, unit_square
from ngsapps.utils import *

order = 3
maxh = 7

# time step and end
# supplementary material says tau=1, but that seems much too high
# tau = 0.001
tau = 1
tend = -1

# diffusion coefficient for rho
DT = 0.004

# determines how quickly the average local swim speed
# decreases with the density
alpha = 0.38

# positive parameters which ensure that, at steady state
# and in the absence of any density or velocity gradients,
# W vanishes everywhere
gamma1 = 0.1
gamma2 = 0.5

# effective diffusion coefficent for W
# ensures continuity of this field
k = 0.1

# self-advection
w1 = 0
# active pressure, >0
w2 = 30

# local propulsion speed
# not sure about v0
# Router: leave space between annulus and domain boundary?
v = AnnulusSpeedCF(Rinner=50, Router=100, phi0=50, vout=0.05, v0=0.2)
vdx = v.Dx()
vdy = v.Dy()
gradv = CoefficientFunction((vdx, vdy))

geo = SplineGeometry()
MakePeriodicRectangle(geo, (-100, -100), (100, 100))
mesh = Mesh(geo.GenerateMesh(maxh=maxh))

fes1 = Periodic(H1(mesh, order=order))
fes = FESpace([fes1, fes1, fes1])

rho, Wx, Wy = fes.TrialFunction()
trho, tWx, tWy = fes.TestFunction()

W = CoefficientFunction((Wx, Wy))
Wxdx = grad(Wx)[0]
Wxdy = grad(Wx)[1]
Wydx = grad(Wy)[0]
Wydy = grad(Wy)[1]
divW = Wxdx + Wydy
gradWx = CoefficientFunction((Wxdx, Wxdy))
gradWy = CoefficientFunction((Wydx, Wydy))

tW = CoefficientFunction((tWx, tWy))
divtW = grad(tWx)[0] + grad(tWy)[1]

g = GridFunction(fes)
grho, gWx, gWy = g.components
gW = CoefficientFunction((gWx, gWy))
vbar = v * exp(-alpha*grho)
gradvbar = gradv*exp(-alpha*grho) - alpha*grad(grho)*vbar*exp(-alpha*grho)
# is this correct?
WdotdelW = CoefficientFunction((gW*gradWx, gW*gradWy))
gradnormWsq = 2*CoefficientFunction((gWx*Wxdx, gWy*Wydy))

# initial values
# grho.Set(exp(-sqr(x)-sqr(y)))
# measure = Integrate(CoefficientFunction(1), mesh)
# grho.Set(CoefficientFunction(1/measure))
grho.Set(CoefficientFunction(1))
gWx.Set(CoefficientFunction(0))
gWy.Set(CoefficientFunction(0))

a = BilinearForm(fes)

# equation for rho
# TODO: boundary terms from partial integration?
# TODO: separate terms which need to be reassembled at every time step
a += SymbolicBFI(-gradvbar*W*trho - vbar*divW*trho - DT*grad(rho)*grad(trho))

# equation for W
a += SymbolicBFI(-0.5*(gradvbar*rho + vbar*grad(rho))*tW - gamma1*W*tW
                 -gamma2*(sqr(gWx)+sqr(gWy))*W*tW - k*divW*divtW
                 -w1*WdotdelW*tW + w2*gradnormWsq*tW)

m = BilinearForm(fes)
m += SymbolicBFI(rho*trho + W*tW)

m.Assemble()

rhs = g.vec.CreateVector()
mstar = m.mat.CreateMatrix()

Draw(vdy, mesh, 'vdy')
Draw(vdx, mesh, 'vdx')
Draw(v, mesh, 'v')
Draw(gW, mesh, 'W')
Draw(grho, mesh, 'rho')

input("Press any key...")
t = 0.0
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

        Redraw(blocking=False)
        # input()
