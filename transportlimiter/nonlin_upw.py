#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:26:14 2017

Solve 
u_t + div(u(1-u)\beta) = 0

@author: pietschm
"""
from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsapps.utils import *
import numpy as np

#from geometries import *
from ngsapps.plotting import *
from limiter import *

ngsglobals.msg_level = 1

order = 1
maxh = 0.05
tau = 0.001
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
aupw = BilinearForm(fes)

# u_t + beta*grad(u) = 0
# aupw += SymbolicBFI((1-2*u)*beta*grad(v)*w)
# aupw += SymbolicBFI(negPart((1-2*u)*beta*n)*v*w, BND, skeleton=True)
# aupw += SymbolicBFI(-(1-2*u)*beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)
# aupw += SymbolicBFI(0.5*abs((1-2*u)*beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# u_t + div(beta*u) = 0
# aupw += SymbolicBFI(-v*(1-u)*beta*grad(w))
# aupw += SymbolicBFI(posPart((1-u)*beta*n)*v*w, BND, skeleton=True)
# aupw += SymbolicBFI((1-u)*beta*n*(v + v.Other())*0.5*(w - w.Other()), skeleton=True)
# aupw += SymbolicBFI(0.5*abs((1-u)*beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# Lax Friedrich
etaf = abs(beta*n)
phi = 0.5*(v*(1-u)*beta*n + v.Other()*(1-u.Other())*beta*n)
phi += etaf*(v-v.Other())

aupw += SymbolicBFI(-v*(1-u)*beta*grad(w))
aupw += SymbolicBFI(phi*(w - w.Other()), skeleton=True)

phib = 0.5*(v*(1-u)*beta*n + 0)
phib += etaf*(v-0)
aupw += SymbolicBFI(phib*(w - w.Other()), BND, skeleton=True)
# aupw += SymbolicBFI(negPart(beta*n)*v*w, BND, skeleton=True)



# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(v*w)

f = LinearForm(fes)
f += SymbolicLFI(0 * w)

print('Assembling aupw...')
aupw.Assemble()
print('Assembling m...')
m.Assemble()
minv = m.mat.Inverse(fes.FreeDofs())
print('Assembling f...')
f.Assemble()

rhs = u.vec.CreateVector()
mstar = aupw.mat.CreateMatrix()

u.Set(0.9*exp(-(x*x+y*y)))

if netgenMesh.dim == 1:
    uplot = Plot(u, mesh=mesh)
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
        
        aupw.Assemble()

        # Explicit
#        aupw.Apply (u.vec, rhs)
#        rhs.data = (-1*tau)*rhs
#        rhs.data += m.mat * u.vec
        rhs.data = m.mat * u.vec - tau * aupw.mat * u.vec
        rhs.data += f.vec
        u.vec.data = minv * rhs
        
        # Implicits
#        rhs.data = m.mat * u.vec
#        mstar.AsVector().data = m.mat.AsVector() + tau * aupw.mat.AsVector()
#        invmat = mstar.Inverse(fes.FreeDofs())
#        u.vec.data = invmat * rhs
        # stabilityLimiter(u, fes, uplot)
        # nonnegativityLimiter(u, fes, uplot)
        # Calculate mass
        print('mass = ' + str(Integrate(u,mesh)))

        if netgenMesh.dim == 1:
            if k % 100 == 0:
                uplot.Redraw()
                plt.pause(0.001)
                input()
        else:
            Redraw(blocking=False)

