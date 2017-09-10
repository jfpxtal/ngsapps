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
import numpy as np

#from geometries import *
from ngsapps.plotting import *
from limiter import *

ngsglobals.msg_level = 0

def abs(x):
    return IfPos(x, x, -x)

def make1DMesh(maxh):
    netmesh = NetMesh()
    netmesh.dim = 1
    start = -5
    L = 10
    N = int(L/maxh)+1
    pnums = []
    for i in range(0, N + 1):
        pnums.append(netmesh.Add(MeshPoint(Pnt(start + L * i / N, 0, 0))))

    for i in range(0, N):
        netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=i+1))
        netmesh.SetMaterial(i+1, 'top'+str(i+1))
        
    netmesh.Add(Element0D(pnums[0], index=1))
    netmesh.Add(Element0D(pnums[N], index=2))
    return netmesh

order = 1
maxh = 0.005
tau = 0.01
tend = -1

del1 = 0.05
del2 = 0.1
D = 0.001

usegeo = "circle"
usegeo = "1d"

if usegeo == "circle":
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=5, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)
elif usegeo == "1d":
    netgenMesh = make1DMesh(maxh)

mesh = Mesh(netgenMesh)
#mesh.Curve(order)

# finite element space
fes = H1(mesh, order=order, dirichlet=[1,2])
v = fes.TrialFunction()
w = fes.TestFunction()

u = GridFunction(fes)
phi = GridFunction(fes)
tmp = GridFunction(fes)
phi_rhs = GridFunction(fes)
q = phi.vec.CreateVector()
q2 = phi.vec.CreateVector()

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# velocity field and absorption
if usegeo == "circle":
    beta = CoefficientFunction((1,0))
elif usegeo == "1d":
    beta = CoefficientFunction(0.1)
    
mu = 0.0

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(v*w)
print('Assembling m...')
m.Assemble()
minv = m.mat.Inverse(fes.FreeDofs())

# Eikonal equation forms
# equation for r


a = BilinearForm(fes)
a += SymbolicBFI(del1*grad(v)*grad(w))
#a += SymbolicBFI(-del1*0.5*(grad(v) + grad(v.Other())) * n * (w - w.Other()), skeleton=True) 
#a += SymbolicBFI(-del1*0.5*(grad(v) + grad(v.Other())) * n * (w - w.Other()), skeleton=True) 
#a += SymbolicBFI(del1*eta / h * (v - v.Other()) * (w - w.Other()), skeleton=True) 
## Dirichlet BSc -- SEEMS TO WORK -- WHY 
#a += SymbolicBFI(-del1*0.5*grad(v) * n * w, BND, skeleton=True) 
#a += SymbolicBFI(-del1*0.5*grad(w) * n * v, BND, skeleton=True) 
#a += SymbolicBFI(del1*eta / h * v * w, BND, skeleton=True) 

# Updwind for Hamilton Jacobi term
aeupw = BilinearForm(fes)
aeupw += SymbolicBFI(grad(phi_rhs)*grad(v)*w)
#aeupw += SymbolicBFI(-grad(phi_rhs)*n*0.5*(v - v.Other())*(w + w.Other()), skeleton=True)
#aeupw += SymbolicBFI(0.5*abs(grad(phi_rhs)*n) * (v - v.Other())*(w - w.Other()), skeleton=True)
#aeupw += SymbolicBFI(IfPos(grad(phi_rhs)*n,0,-grad(phi_rhs)*n)*v*w, BND, skeleton=True)

feik = LinearForm(fes)
#f += SymbolicLFI( -(gradrho[0]*gradrho[0] + gradrho[1]*gradrho[1]) * phi + 1*phi) # 
feik += SymbolicLFI(1/((1-u)*(1-u)+del2)*w) # FIXME
#feik += SymbolicLFI(1*w)

feik.Assemble()
a.Assemble()

mstar = m.mat.CreateMatrix()

if netgenMesh.dim == 1:
    phiplot = Plot(phi, mesh=mesh)
    plt.show(block=False)
else:
    Draw(phi, mesh, 'phi')


def EikonalSolver():
    tau = 0.4
    k = 0
    errNewton = 1e99
    with TaskManager():
        while errNewton > 1e-6:
            k += 1
            
            # Apply nonlinear operator
            phi_rhs.vec.data = phi.vec
            aeupw.Assemble()
            a.Apply(phi.vec, q)
            aeupw.Apply(phi.vec, q2)
            q.data += q2
            q.data -= feik.vec
            
            tmp.vec.data = q2
            
            # Linearized HJ-Operator - has additional 2 from square
            aeupw.Assemble()
            mstar.AsVector().data = a.mat.AsVector() + 2*aeupw.mat.AsVector()
            
            # Solve for update and perform Newton step        
            invmat = mstar.Inverse(fes.FreeDofs())
            
            q2.data = invmat * q
            phi.vec.data -= tau*(q2)
            
            errNewton = q2.Norm()
            
            print('Res error = ' + str(q2.Norm()) + '\n') # L2norm of update
    
            if netgenMesh.dim == 1:
                if k % 50 == 0:
                    phiplot.Redraw()
                    plt.pause(0.001)
#            else:
#                Redraw(blocking=False)
        
#            input("")



# upwind fluxes scheme
aupw = BilinearForm(fes)

# u_t + beta*grad(u) = 0
#aupw += SymbolicBFI(-(1-2*u)*grad(phi)*grad(v)*w)
#aupw += SymbolicBFI( IfPos((1-2*u)*grad(phi)*n,(1-2*u)*grad(phi)*n,0)*v*w, BND, skeleton=True)
#aupw += SymbolicBFI((1-2*u)*grad(phi)*n* (v - v.Other())*0.5*(w + w.Other()), skeleton=True)
#aupw += SymbolicBFI(0.5*abs((1-2*u)*grad(phi)*n)*(v - v.Other())*(w - w.Other()), skeleton=True)

# u_t + div(beta*u) = 0
aupw += SymbolicBFI(-v*(1-u)*grad(phi)*grad(w))
#aupw += SymbolicBFI(IfPos((1-u)*beta*n,(1-u)*beta*n,0)*v*w, BND, skeleton=True)
#aupw += SymbolicBFI((1-u)*beta*n* (v + v.Other())*0.5*(w - w.Other()), skeleton=True)
#aupw += SymbolicBFI(0.5*abs((1-u)*beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)


# Diffusion
#D = 0.001 # Diffusion coefficient
#eta = 3 # Penalty parameter
asip = BilinearForm(fes)
asip += SymbolicBFI(D*grad(v)*grad(w))
#asip += SymbolicBFI(-D*0.5*(grad(v)+grad(v.Other())) * n * (w - w.Other()), skeleton=True)
#asip += SymbolicBFI(-D*0.5*(grad(w)+grad(w.Other())) * n * (v - v.Other()), skeleton=True)
#asip += SymbolicBFI(D*eta / h * (v - v.Other()) * (w - w.Other()), skeleton=True)
    
# SEEMS TO WORK -- WHY 
#asip += SymbolicBFI(-D*0.5*(grad(v)) * n * (w), BND, skeleton=True) #, definedon=topMat)
#asip += SymbolicBFI(-D*0.5*(grad(w)) * n * (v), BND, skeleton=True) #, definedon=topMat)
#asip += SymbolicBFI(D*eta / h * (v) * w, BND, skeleton=True) #, definedon=topMat)
    


# Lax Friedrich
#etaf = abs(beta*n)
#phi = 0.5*(v*(1-v)*beta*n + v.Other()*(1-v.Other())*beta*n)
#phi += etaf*(v-v.Other())
#
#aupw += SymbolicBFI(-v*(1-v)*beta*grad(w))
#aupw += SymbolicBFI(phi*(w - w.Other()), skeleton=True)
#
#phib = 0.5*(v*(1-u)*beta*n + 0)
#phib += etaf*(v-0)
#aupw += SymbolicBFI(phib*(w - w.Other()), BND, skeleton=True)
#aupw += SymbolicBFI( IfPos(-beta*n,-beta*n,0)*v*w, BND, skeleton=True)





f = LinearForm(fes)
f += SymbolicLFI(0 * w)

print('Assembling aupw...')
aupw.Assemble()
print('Assembling asip...')
asip.Assemble()

print('Assembling f...')
f.Assemble()

rhs = u.vec.CreateVector()
mstar = aupw.mat.CreateMatrix()

xshift = 3
u.Set(0.9*exp(-((x-xshift)*(x-xshift)+y*y)))

if netgenMesh.dim == 1:
    plt.figure()
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
        
        # Solve Eikonal equations
        feik.Assemble()
        EikonalSolver()
        
        #input("")
        
        aupw.Assemble()

        # Explicit
#        aupw.Apply (u.vec, rhs)
#        rhs.data = (-1*tau)*rhs
#        rhs.data += m.mat * u.vec
#        rhs.data = m.mat * u.vec - tau * (aupw.mat * u.vec) # + asip.mat * u.vec)
#        rhs.data += f.vec
#        u.vec.data = minv * rhs
        
        # Implicits
        rhs.data = m.mat * u.vec
        mstar.AsVector().data = m.mat.AsVector() + tau * (asip.mat.AsVector() + aupw.mat.AsVector())
        invmat = mstar.Inverse(fes.FreeDofs())
        u.vec.data = invmat * rhs
        
        if netgenMesh.dim == 1:
            stabilityLimiter(u, fes, uplot)
            nonnegativityLimiter(u, fes, uplot)
            
        # Calculate mass
        print('mass = ' + str(Integrate(u,mesh)))

        if netgenMesh.dim == 1:
            if k % 1 == 0:
                uplot.Redraw()
                phiplot.Redraw()
                plt.pause(0.001)
        else:
            Redraw(blocking=False)
            
#        input("")
