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

order = 3
maxh = 0.01
tau = 0.01
tend = -1

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
fes1 = L2(mesh, order=order)
fes = FESpace([fes1, fes1], flags={'dgjumps': True})

v1, v2 = fes.TrialFunction()
w1, w2 = fes.TestFunction()

[u1,u2] = GridFunction(fes)
#uc = GridFunction(fes)

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# velocity field and absorption
if usegeo == "circle":
    beta = CoefficientFunction((1,0))
elif usegeo == "1d":
    beta = CoefficientFunction(0.1)

mu = 0.0

# upwind fluxes scheme
a = BilinearForm(fes)

a += SymbolicBFI((-v1*(1-v1)*beta - sqrt(D)v2)*grad(w1))
a += SymbolicBFI(v2*w2 - sqrt(D)*grad(v2)*grad(w2))


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
phi = 0.5*(v1*(1-v1) + v1.Other(0)*(1-v1.Other(0)))*beta*n
phi += 0.5*etaf*(v1-v1.Other(0))

phiR = IfPos(beta*n, beta*n*IfPos(v1-0.5, 0.25, v1*(1-v1)), 0)

#a += SymbolicBFI(-v*(1-v)*beta*grad(w))
# Convective flux
a += SymbolicBFI(phi*(w1 - w1.Other()), VOL, skeleton=True)
a += SymbolicBFI(phiR*w1, BND, skeleton=True)



# semi-implicit (not working yet, GridFunction.Other())
# etaf = abs(beta*n)
# phi = 0.5*(v*(1-u) + v.Other(0)*(1-u.Other(0)))*beta*n
# phi += 0.5*etaf*(v-v.Other(0))

# a += SymbolicBFI(-v*(1-u)*beta*grad(w))
# a += SymbolicBFI(phi*(w - w.Other()), skeleton=True)
# a += SymbolicBFI(phi*w, BND, skeleton=True) FIXME


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
# mstar = m.mat.CreateMatrix()

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
    a.Apply(y, rhs)
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

        # a.Apply(u.vec, rhs)
        # fes.SolveM(rho=CoefficientFunction(1), vec=rhs)
        # u.vec.data -= tau*rhs

        # Implicits
        # rhs.data = m.mat * u.vec
        # mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
        # invmat = mstar.Inverse(fes.FreeDofs())
        # u.vec.data = invmat * rhs

        stabilityLimiter(u, fes, uplot)
        nonnegativityLimiter(u, fes, uplot)

        # Calculate mass
        print('mass = ' + str(Integrate(u,mesh)))

        if netgenMesh.dim == 1:
            if k % 150 == 0:
                uplot.Redraw()
                plt.pause(0.001)
                # input()
        else:
            Redraw(blocking=False)


