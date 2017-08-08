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

ngsglobals.msg_level = 1

order = 1
maxh = 0.01
tau = 0.001
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
fes = L2(mesh, order=order, flags={'dgjumps': True})
v = fes.TrialFunction()
w = fes.TestFunction()

u = GridFunction(fes)

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

# # centered trial
# a += SymbolicBFI((1-2*u)*beta*grad(v)*w)
# a += SymbolicBFI(negPart((1-2*u)*beta*n)*v*w, BND, skeleton=True)
# a += SymbolicBFI(-(1-2*u)*beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)

# # centered test
# a += SymbolicBFI(-(1-2*u)*beta*v*grad(w))
# a += SymbolicBFI(negPart((1-2*u)*beta*n)*v*w, BND, skeleton=True)
# a += SymbolicBFI(-(1-2*u)*beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)

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
etaf = abs(beta*n)
phi = 0.5*(v*(1-v) + v.Other(0)*(1-v.Other(0)))*beta*n
phi += 0.5*etaf*(v-v.Other(0))

a += SymbolicBFI(-v*(1-v)*beta*grad(w))
a += SymbolicBFI(phi*(w - w.Other()), skeleton=True)
a += SymbolicBFI(phi*w, BND, skeleton=True)


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

if netgenMesh.dim == 1:
    uplot = Plot(u, mesh=mesh, subdivision=3)
    plt.show(block=False)
else:
    Draw(u, mesh, 'u')

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
        a.Apply(u.vec, rhs)
        fes.SolveM(rho=CoefficientFunction(1), vec=rhs)
        u.vec.data -= tau*rhs
        # rhs.data = (-1*tau)*rhs
        # rhs.data += m.mat * u.vec
        # rhs.data = m.mat * u.vec - tau * a.mat * u.vec
        # rhs.data += f.vec
        # u.vec.data = minv * rhs

        # Implicits
#        rhs.data = m.mat * u.vec
#        mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
#        invmat = mstar.Inverse(fes.FreeDofs())
#        u.vec.data = invmat * rhs

        # stabilityLimiter(u, fes, uplot)
        # nonnegativityLimiter(u, fes, uplot)

        # Calculate mass
        print('mass = ' + str(Integrate(u,mesh)))

        if netgenMesh.dim == 1:
            if k % 10 == 0:
                uplot.Redraw()
                plt.pause(0.001)
                input()
        else:
            Redraw(blocking=False)


