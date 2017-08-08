#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:26:14 2017

Solve 
u_t + div(u(1-u)\beta) = 0

@author: pietschm
"""
from netgen.geom2d import SplineGeometry
from ngsolve import *
import numpy as np

from ngsapps.utils import *
from ngsapps.plotting import *
from limiter import *
from rungekutta import *

ngsglobals.msg_level = 1

order = 1
maxh = 0.02
tau = 0.0005
tend = -1


usegeo = "circle" 
usegeo = "1d"

if usegeo == "circle":
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=1, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)
elif usegeo == "1d":
    netgenMesh = Make1DMesh(-1, 1, maxh)

mesh = Mesh(netgenMesh)
#mesh.Curve(order)

# finite element space
fes = L2(mesh, order=order, flags={'dgjumps': True})
v = fes.TrialFunction()
w = fes.TestFunction()

u = GridFunction(fes)
uc = GridFunction(fes)

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# velocity field and absorption
if usegeo == "circle":
    beta = CoefficientFunction((1,0))
elif usegeo == "1d":
    beta = CoefficientFunction(1)

mu = 0.0

# upwind fluxes scheme
a = BilinearForm(fes)


# u_t + beta*grad(u) = 0
# a += SymbolicBFI((1-2*u)*beta*grad(v)*w)
# a += SymbolicBFI(negPart((1-2*u)*beta*n)*v*w, BND, skeleton=True)
# a += SymbolicBFI(-(1-2*u)*beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)
# a += SymbolicBFI(0.5*abs((1-2*u)*beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# u_t + div(beta*u) = 0
# a += SymbolicBFI(-v*(1-2*u)*beta*grad(w))
# a += SymbolicBFI(posPart((1-2*u)*beta*n)*v*w, BND, skeleton=True)
# a += SymbolicBFI((1-2*u)*beta*n*(v + v.Other())*0.5*(w - w.Other()), skeleton=True)
# a += SymbolicBFI(0.5*10*abs((1-2*u)*beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# Lax Friedrichs
# explicit
etaf = abs(beta*n)
phi = 0.5*(v*(1-v) + v.Other(0)*(1-v.Other(0)))*beta*n
phi += 0.5*etaf*(v-v.Other(0))

phiR = IfPos(beta*n, beta*n*IfPos(v-0.5, 0.25, v*(1-v)), 0)

a += SymbolicBFI(-v*(1-v)*beta*grad(w))
a += SymbolicBFI(phi*(w - w.Other()), VOL, skeleton=True)
#a += SymbolicBFI(phiR*w, BND, skeleton=True)

# semi-implicit (not working yet, GridFunction.Other())
# etaf = abs(beta*n)
# phi = 0.5*(v*(1-u) + v.Other(0)*(1-u.Other(0)))*beta*n
# phi += 0.5*etaf*(v-v.Other(0))

# a += SymbolicBFI(-v*(1-u)*beta*grad(w))
# a += SymbolicBFI(phi*(w - w.Other()), skeleton=True)
# a += SymbolicBFI(phi*w, BND, skeleton=True) FIXME

# Add diffusion
D = 0.05
eta = 3
asip = BilinearForm(fes)
asip += SymbolicBFI(D*grad(v)*grad(w))
asip += SymbolicBFI(-D*0.5*(grad(v)+grad(v.Other())) * n * (w - w.Other()), skeleton=True)
asip += SymbolicBFI(-D*0.5*(grad(w)+grad(w.Other())) * n * (v - v.Other()), skeleton=True)
asip += SymbolicBFI(D*eta / h * (v - v.Other()) * (w - w.Other()), skeleton=True)
    
Dirichlet = True
if Dirichlet:
    asip += SymbolicBFI(-D*0.5*(grad(v)) * n * (w), BND, skeleton=True) #, definedon=topMat)
    asip += SymbolicBFI(-D*0.5*(grad(w)) * n * (v), BND, skeleton=True) #, definedon=topMat)
    asip += SymbolicBFI(D*eta / h * (v) * w, BND, skeleton=True) #, definedon=topMat)
    
    
asip.Assemble()
    
# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(v*w)

f = LinearForm(fes)
f += SymbolicLFI(0 * w)

# print('Assembling a...')
# a.Assemble()
print('Assembling m...')
m.Assemble()
minv = m.mat.Inverse(fes.FreeDofs())
# print('Assembling f...')
# f.Assemble()

rhs = u.vec.CreateVector()
rhs2 = u.vec.CreateVector()
mstar = m.mat.CreateMatrix()

u.Set(0.9*exp(-2*(x*x+y*y)))
uc.Set(0.5+0*x)
# u.Set(CoefficientFunction(0.4))

if netgenMesh.dim == 1:
    uplot = Plot(u, mesh=mesh, subdivision=3)
    ucplot = Plot(uc, mesh=mesh, subdivision=3)
    plt.axis([-1, 1, 0, 1])
    plt.show(block=False)
else:
    Draw(u, mesh, 'u')

def step(t, y):
    a.Apply(y,rhs)
    asip.Apply(y, rhs2)
    rhs.data += rhs2
#    rhs.data = asip.mat * y
    fes.SolveM(rho=CoefficientFunction(1), vec=rhs)
    return -rhs

input("Press any key...")

# Explicit Euler
t = 0.0
k = 0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
        print("\nt = {:10.6e}".format(t))
        t += tau
        k += 1

        # a.Assemble()

        # Explicit
        # u.vec.data = RungeKutta(euler, tau, step, t, u.vec)
        # TODO: limit after each interior Euler step!
        u.vec.data = RungeKutta(rk4, tau, step, t, u.vec)

#        asip.Apply(u.vec, rhs)
#        fes.SolveM(rho=CoefficientFunction(1), vec=rhs)
#        u.vec.data -= tau*rhs

        # Implicits
#        rhs.data = m.mat * u.vec
#        mstar.AsVector().data = m.mat.AsVector() + tau * asip.mat.AsVector()
#        invmat = mstar.Inverse(fes.FreeDofs())
#        u.vec.data = invmat * rhs

        stabilityLimiter(u, fes, uplot)
        nonnegativityLimiter(u, fes, uplot)

        # Calculate mass
        print('mass = ' + str(Integrate(u,mesh)))

        if netgenMesh.dim == 1:
            if k % 50 == 0:
                uplot.Redraw()
                plt.pause(0.001)
                # input()
        else:
            Redraw(blocking=False)


