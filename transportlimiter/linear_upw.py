#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:26:14 2017

Solve 
u_t + div(u\beta) = 0

@author: pietschm
"""
from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsapps.utils import *
import numpy as np

from ngsapps.plotting import *

ngsglobals.msg_level = 1

order = 1
maxh = 0.1
tau = 0.005
tend = -1

usegeo = "circle"
usegeo = "1d"
usegeo = "test"

if usegeo == "circle":
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=1, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)
elif usegeo == "1d":
    netgenMesh = Make1DMesh(-1, 1, maxh)
elif usegeo == "test":
    netgenMesh = NetMesh()
    netgenMesh.dim = 1
    start = -1
    L = 2
    N = 3 # Nof elements ?
    pnums = []
    for i in range(0, N + 1):
        pnums.append(netgenMesh.Add(MeshPoint(Pnt(start + L * i / N, 0, 0))))

    for i in range(0, N):
        netgenMesh.Add(Element1D([pnums[i], pnums[i + 1]]))
        # netmesh.SetMaterial(i+1, 'top'+str(i+1))
        # netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))

#    netgenMesh.Add(Element0D(pnums[0], index=1))
#    netgenMesh.Add(Element0D(pnums[N], index=2))
    #netmesh.SetMaterial(1, 'top')
    #return netmesh

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
elif usegeo == "1d" or usegeo == "test":
    beta = CoefficientFunction(1)
    
mu = 0.0

# upwind fluxes scheme
aupw = BilinearForm(fes)
#sdfds
# u_t + beta*grad(u) = 0
aupw += SymbolicBFI(beta*grad(v)*w)
aupw += SymbolicBFI(negPart(beta*n)*v*w, BND, skeleton=True)
aupw += SymbolicBFI(-beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)
aupw += SymbolicBFI(0.5*abs(beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# centered flux
# aupw += SymbolicBFI(beta*grad(v)*w)
# aupw += SymbolicBFI(negPart(beta*n)*v*w, BND, skeleton=True)
# aupw += SymbolicBFI(-beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)

# u_t + div(beta*u) = 0
# aupw += SymbolicBFI(-v*beta*grad(w))
# aupw += SymbolicBFI(posPart(beta*n)*v*w, BND, skeleton=True)
# aupw += SymbolicBFI(beta*n*(v + v.Other())*0.5*(w - w.Other()), skeleton=True)
# aupw += SymbolicBFI(0.5*abs(beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# centered flux
# aupw += SymbolicBFI(-v*grad(w))
# aupw += SymbolicBFI(posPart(n)*v*w, BND, skeleton=True)
# aupw += SymbolicBFI(n*(v + v.Other())*0.5*(w - w.Other()), skeleton=True)


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

if usegeo == "test":
    u.vec[2] = 1
    u.vec[3] = 1
else:
    u.Set(exp(-(x*x+y*y)))

if netgenMesh.dim == 1:
    uplot = Plot(u, mesh=mesh)
    plt.show(block=False)
else:
    Draw(u, mesh, 'u')

input("Press any key...")

# Explicit Euler
t = 0.0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
        print("\nt = {:10.6e}".format(t))
        t += tau

        # Explicit
        rhs.data = m.mat * u.vec - tau * aupw.mat * u.vec
#        rhs.data += f.vec
        u.vec.data = minv * rhs
        
        # Implicit
#        rhs.data = m.mat * u.vec
#        mstar.AsVector().data = m.mat.AsVector() + tau * aupw.mat.AsVector()
#        invmat = mstar.Inverse(fes.FreeDofs())
#        u.vec.data = invmat * rhs

        # Calculate mass
        print('mass = ' + str(Integrate(u,mesh)))

        if netgenMesh.dim == 1:
            uplot.Redraw()
            plt.pause(0.05)
            input()
        else:
            Redraw(blocking=False)
