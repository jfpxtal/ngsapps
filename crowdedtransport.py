# Flow Characteristics in a Crowded Transport Model
# https://arxiv.org/abs/1502.02715

from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.comp import Region
import matplotlib.pyplot as plt
from ngsapps.utils import *

order = 3
conv_order = 6
maxh = 0.15

vtkoutput = False

# diffusion coefficient
D = 0.01
# inflow rates
alpha1 = 0.2
alpha2 = 0.4
# outflow rates
beta1 = 0.4
beta2 = 0.2

# velocity field
V = CoefficientFunction(x)
u = CoefficientFunction((1.0, 0.0))

# convolution kernels
# gaussian:
thin = 200
k0 = 5
# k0 = 1
K = k0*exp(-thin*(x*x+y*y))
# partial derivatives
# Kdx = -2*k0*thin*x*exp(-thin*(x*x+y*y))
# Kdy = -2*k0*thin*y*exp(-thin*(x*x+y*y))

# compact support:
# norm = sqrt(x*x+y*y)
# width = 8
# K = IfPos(1-norm/width, 1-norm/width, 0)
# Kdx = IfPos(1-norm/width, -x/norm/width, 0)
# Kdy = IfPos(1-norm/width, -y/norm/width, 0)

# time step and end
tau = 0.01
tend = -1

# jump penalty in asip below
# eta = 100
# eta = 50
eta = 40

geo = SplineGeometry()
# geo.AddRectangle((0, 0), (2, 1), bcs=[1, 2, 3, 4])

# rectangular corridor with two entrances to the left
# and two exits to the right
#         _________________
#        |                 |
# alpha2                     beta2
#        |                 |
# alpha1                     beta1
#        |_________________|
pts = []
ycoords = [0, 0.15, 0.35, 0.65, 0.85, 1]
for y in reversed(ycoords):
    pts.append(geo.AppendPoint(0, y))
for y in ycoords:
    pts.append(geo.AppendPoint(2, y))

bcs = [None, 'alpha2', None, 'alpha1', None, None, None, 'beta1', None, 'beta2', None, None]
for i, (pt1, pt2) in enumerate(zip(pts, pts[1:]+[pts[0]])):
    geo.Append(['line', pt1, pt2], bc=bcs[i], leftdomain=1)

def MakePolygon(geo, pts, **args):
    ptids = [geo.AppendPoint(*p) for p in pts]
    for p1, p2 in zip(ptids, ptids[1:]+[ptids[0]]):
        geo.Append(['line', p1, p2], **args)

# obstacles
# geo.AddCircle((1.5, 0.5), 0.2, leftdomain=0, rightdomain=1)
# MakePolygon(geo, [(1.3, 0.35), (1.6, 0.35), (1.6, 0.65)], leftdomain=0, rightdomain=1)
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
mesh.Curve(order)

# finite element space
fes = L2(mesh, order=order, flags={'dgjumps': True})
rho = fes.TrialFunction()
phi = fes.TestFunction()

# initial value
rho2 = GridFunction(fes)
rho2.Set(CoefficientFunction(0.5))
# rho2.Set(CoefficientFunction(0.0))

# set up coefficient functions for convolution terms
# and caches to prevent unnecessary calculation in aconv.Assemble()
# convx = Convolve(rho2, Kdx, mesh, conv_order)
# convy = Convolve(rho2, Kdy, mesh, conv_order)
conv = Convolve(rho2, K, mesh, conv_order)
# convx = Convolve(grad(rho2)[0], K, mesh, conv_order)
# convy = Convolve(grad(rho2)[1], K, mesh, conv_order)
# convx_cache = Cache(convx, fes)
# convy_cache = Cache(convy, fes)

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# symmetric interior penalty method
# for the diffusion
asip = BilinearForm(fes)
asip += SymbolicBFI(D*grad(rho)*grad(phi))
asip += SymbolicBFI(-D*0.5*(grad(rho)+grad(rho.Other())) * n * (phi - phi.Other()), skeleton=True)
asip += SymbolicBFI(-D*0.5*(grad(phi)+grad(phi.Other())) * n * (rho - rho.Other()), skeleton=True)
asip += SymbolicBFI(D*eta / h * (rho - rho.Other()) * (phi - phi.Other()), skeleton=True)

# boundary terms
aF = BilinearForm(fes)
aF += SymbolicBFI(alpha1*rho*phi, definedon=Region(mesh, BND, 'alpha1'), skeleton=True)
aF += SymbolicBFI(alpha2*rho*phi, definedon=Region(mesh, BND, 'alpha2'), skeleton=True)
aF += SymbolicBFI(beta1*rho*phi, definedon=Region(mesh, BND, 'beta1'), skeleton=True)
aF += SymbolicBFI(beta2*rho*phi, definedon=Region(mesh, BND, 'beta2'), skeleton=True)

def abs(x):
    return IfPos(x, x, -x)

g = GridFunction(fes)
# convolution term
aconv = BilinearForm(fes)
aconv += SymbolicBFI(-rho*(1-rho2)*grad(g)*grad(phi))
aconv += SymbolicBFI((1-rho2)*grad(g)*n*0.5*(rho+rho.Other())*(phi-phi.Other()), skeleton=True)
aconv += SymbolicBFI(0.5*abs((1-rho2)*grad(g)*n) * (rho - rho.Other())*(phi - phi.Other()), skeleton=True)
# convbfi = SymbolicBFI_NoDG(-rho*(1-rho2)*CoefficientFunction((convx_cache, convy_cache))*grad(phi))
# convbfi = SymbolicBFI_NoDG(-rho*(1-rho2)*CoefficientFunction((gx, gy))*grad(phi))
# convbfi = SymbolicBFI_NoDG(-rho*CoefficientFunction((convx, convy))*grad(phi))
# aconv += convbfi

# convx_cache.SetBFI(convbfi)
# convy_cache.SetBFI(convbfi)


# upwind scheme for the advection
aupw = BilinearForm(fes)
aupw += SymbolicBFI(-rho*(1-rho2)*u*grad(phi))
aupw += SymbolicBFI((1-rho2)*u*n*0.5*(rho+rho.Other())*(phi-phi.Other()), skeleton=True)
aupw += SymbolicBFI(0.5*abs((1-rho2)*u*n) * (rho - rho.Other())*(phi - phi.Other()), skeleton=True)

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(rho*phi)

f = LinearForm(fes)
f += SymbolicLFI(alpha1 * phi, definedon=Region(mesh, BND, 'alpha1'), skeleton=True)
f += SymbolicLFI(alpha2 * phi, definedon=Region(mesh, BND, 'alpha2'), skeleton=True)

print('Assembling asip...')
asip.Assemble()
print('Assembling aF...')
aF.Assemble()
print('Assembling m...')
m.Assemble()
print('Assembling f...')
f.Assemble()

rhs = rho2.vec.CreateVector()
mstar = asip.mat.CreateMatrix()

Draw(g, mesh, 'conv')
Draw(rho2, mesh, 'rho')

times = [0.0]
entropy = ZLogZCF(rho2) - rho2*V + ZLogZCF(1-rho2) + rho2*conv
ents = [Integrate(entropy, mesh)]
fig, ax = plt.subplots()
line, = ax.plot(times, ents)
plt.show(block=False)

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[rho2],names=["rho"],filename="crowdtrans/oneminusrho/oneminusrho",subdivision=3)
    vtk.Do()

# input("Press any key...")
# semi-implicit Euler
t = 0.0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
        print("\nt = {:10.6e}".format(t))
        t += tau

        # print('Assembling aupw...')
        aupw.Assemble()
        # print('Calculating convolution integrals...')
        conv.CacheCF()
        g.Set(conv)
        # print('Assembling aconv...')
        aconv.Assemble()

        rhs.data = m.mat * rho2.vec
        rhs.data += tau * f.vec

        mstar.AsVector().data = m.mat.AsVector() + tau * (asip.mat.AsVector() + aF.mat.AsVector() + aupw.mat.AsVector() + aconv.mat.AsVector())
        # mstar.AsVector().data = m.mat.AsVector() + tau * (asip.mat.AsVector() + aF.mat.AsVector() + aupw.mat.AsVector())
        invmat = mstar.Inverse(fes.FreeDofs())
        rho2.vec.data = invmat * rhs

        Redraw(blocking=False)
        times.append(t)
        ents.append(Integrate(entropy, mesh))
        line.set_xdata(times)
        line.set_ydata(ents)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

        if vtkoutput:
            vtk.Do()

        # if t > 0.75:
        #     input()
