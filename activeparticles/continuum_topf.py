from ngsolve import *
from netgen.geom2d import SplineGeometry, unit_square
from ngsapps.utils import *
import settings
import pickle
from datetime import datetime
import re, os

ngsglobals.msg_level = 1

# FIXME: AnnulusSpeedCF Dx, Dy do not respect smear

order = 3
maxh = 7

vtkoutput = False

# time step and end
# tau = 0.01
# tau = 0.5
tau = 5
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
# k = 0.1
k = 2

# self-advection
w1 = 0
# active pressure, >0
w2 = 30

mesh, v, vdx, vdy = settings.annulusInPeriodicSquare(order, maxh)
# mesh, v, vdx, vdy = settings.annulus(order, maxh)
# mesh, v, vdx, vdy = settings.singleSawtooth(order, 2)

gradv = CoefficientFunction((vdx, vdy))

fesRho = Periodic(H1(mesh, order=order))
fesW = Periodic(H1(mesh, order=order-1))
fes = FESpace([fesRho, fesW, fesW])

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
tWxdx = grad(tWx)[0]
tWxdy = grad(tWx)[1]
tWydx = grad(tWy)[0]
tWydy = grad(tWy)[1]
divtW = tWxdx + tWydy
gradtWx = CoefficientFunction((tWxdx, tWxdy))
gradtWy = CoefficientFunction((tWydx, tWydy))

g = GridFunction(fes)
grho, gWx, gWy = g.components
gW = CoefficientFunction((gWx, gWy))
vbar = v * exp(-alpha*grho)
gradvbar = gradv*exp(-alpha*grho) - alpha*grad(grho)*vbar
WdotdelW = CoefficientFunction((gW*gradWx, gW*gradWy))
gradnormWsq = 2*gWx*gradWx + 2*gWy*gradWy

# initial values
# grho.Set(exp(-sqr(x)-sqr(y)))
# measure = Integrate(CoefficientFunction(1), mesh)
# grho.Set(CoefficientFunction(1/measure))
grho.Set(CoefficientFunction(1))
# grho.Set(CoefficientFunction(0.01))
gWx.Set(CoefficientFunction(0))
gWy.Set(CoefficientFunction(0))

a = BilinearForm(fes)

# equation for rho
# TODO: boundary terms from partial integration?
# TODO: separate terms which need to be reassembled at every time step
a += SymbolicBFI(vbar*W*grad(trho) - DT*grad(rho)*grad(trho))
# a += SymbolicBFI((vbar*W*grad(trho) - DT*grad(rho)*grad(trho)).Compile(True))
# a += SymbolicBFI(-gradvbar*W*trho - vbar*divW*trho - DT*grad(rho)*grad(trho))

# # equation for W
a += SymbolicBFI(0.5*vbar*rho*divtW - gamma1*W*tW
                 -gamma2*(sqr(gWx)+sqr(gWy))*W*tW - k*divW*divtW
                 -w1*WdotdelW*tW + w2*gradnormWsq*tW)
# a += SymbolicBFI((0.5*vbar*rho*divtW - gamma1*W*tW
#                  -gamma2*(sqr(gWx)+sqr(gWy))*W*tW - k*divW*divtW
#                  -w1*WdotdelW*tW + w2*gradnormWsq*tW).Compile(True))
# a += SymbolicBFI(-0.5*(gradvbar*rho + vbar*grad(rho))*tW - gamma1*W*tW
#                  -gamma2*(sqr(gWx)+sqr(gWy))*W*tW - k*gradWx*gradtWx - k*gradWy*gradtWy
#                  -w1*WdotdelW*tW + w2*gradnormWsq*tW)

m = BilinearForm(fes)
m += SymbolicBFI(rho*trho + W*tW)

m.Assemble()

rhs = g.vec.CreateVector()
mstar = m.mat.CreateMatrix()

t = 0

path = './'
files = next(os.walk(path))[2]
reg = re.compile('vout0_t=(\d+)_wall=\d+:\d+\.dat\Z')
files = map(lambda f: (int(reg.search(f).group(1)), f), filter(reg.search, files))
files = sorted(files, key=lambda f: f[0])
print(files)
files = map(lambda f: (f[0], pickle.load(open(path+f[1], 'rb')).tolist()), files)

t, g.vec.FV().NumPy()[:] = list(files)[-1]

Draw(vdy, mesh, 'vdy')
Draw(vdx, mesh, 'vdx')
Draw(v, mesh, 'v')
Draw(gW, mesh, 'W')
Draw(vbar*gW-DT*grad(grho), mesh,'J')
Draw(grho, mesh, 'rho')

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh, coefs=[g.components[0], g.components[1], g.components[2]],names=["rho", "Wx", "Wy"], filename="instab/instab",subdivision=3)
    vtk.Do()

input("Press any key...")
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

        now = datetime.now()
        if t % 5000 == 0:
            pickle.dump(g.vec.FV().NumPy(), open('vout0_t={}_wall={:%H:%M}.dat'.format(t, now), 'wb'))

        # if now.hour > 7:
        #     break

        Redraw(blocking=False)

        if vtkoutput:
            vtk.Do()
