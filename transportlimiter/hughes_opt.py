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
import time


#from geometries import *
from ngsapps.utils import *
from ngsapps.plotting import *
from limiter import *
from DGForms import *
from rungekutta import *

import pickle

ngsglobals.msg_level = 0

def abs(x):
    return IfPos(x, x, -x)

order = 1
maxh = 0.03
tau = 0.02
tend = 3.5
times = np.linspace(0.0,tend,np.ceil(tend/tau)) # FIXME: make tend/tau integer
vtkoutput = False

del1 = 0.1 # Regularization parameters
del2 = 0.1
D = 0.05

Na = 1 # Number of agents
width = 1 # width of conv kernel
alpha = 0.1 # Regularization parameter
cK = 0.1 # Parameter to control strength of attraction

vels = np.zeros((Na,times.size)) # Position of agents

# Convolution kernel
# compact support:
#K = 0*x
#norm = sqrt((x-agents[0])*(x-agents[0])+(y)*(y))
#norm = sqrt((x)*(x)+y*y)

#K = IfPos(1-norm/width, 1-norm/width, 0)



eta = 5 # Penalty parameter

usegeo = "circle"
usegeo = "1d"

if usegeo == "circle":
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=5, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)
elif usegeo == "1d":
    #netgenMesh = make1DMesh(maxh)
    netgenMesh = Make1DMesh(-5, 5, maxh)

mesh = Mesh(netgenMesh)

# finite element space
fes = L2(mesh, order=order, flags={'dgjumps': True})
v = fes.TrialFunction()
w = fes.TestFunction()

# Gridfunctions
u = GridFunction(fes)
g = GridFunction(fes)
gadj = GridFunction(fes)
phi = GridFunction(fes)

lam1 = GridFunction(fes)
lam2 = GridFunction(fes)

tmp = GridFunction(fes)
phi_rhs = GridFunction(fes)
q = phi.vec.CreateVector()
q2 = phi.vec.CreateVector()

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(v*w)
print('Assembling m...')
m.Assemble()
minv = m.mat.Inverse(fes.FreeDofs())

# Eikonal equation forms
a = SIPForm(del1, eta, fes, v, w, h, n, Dirichlet=True)
aeupw = UpwindFormNonDivergence(fes, grad(phi_rhs), v, w, h, n, Compile=True)# Updwind for Hamilton Jacobi term
feik = LinearForm(fes)
feik += SymbolicLFI(1/((1-u)*(1-u)+del2)*w) # FIXME

feik.Assemble()
a.Assemble()
mstar = m.mat.CreateMatrix()
a.Apply(phi.vec, q)

# Solve reg. Eikonal equation usign Newton's scheme
def EikonalSolver():
    tau = 1
    k = 0
    errNewton = 1e99
    with TaskManager():
        while errNewton > 1e-6:
            k += 1
            
            # Apply nonlinear operator
            phi_rhs.vec.data = phi.vec
            #aeupw.Assemble()
            #a.Apply(phi.vec, q)
            q.data = a.mat*phi.vec
            aeupw.Apply(phi.vec, q2)
            q.data += q2
            q.data -= feik.vec
            
            # Linearized HJ-Operator - has additional 2 from square
            aeupw.Assemble()
            mstar.AsVector().data = a.mat.AsVector() + 2*aeupw.mat.AsVector()
            
            # Solve for update and perform Newton step        
            invmat = mstar.Inverse(fes.FreeDofs())
            phi.vec.data -= invmat * q   

            errNewton = q.Norm()
        print('Newton finished with Res error = ' + str(q.Norm()) + ' after ' + str(k) + 'steps \n') # L2norm of update

# Forms for Hughes Model diff-transport eq
#aupw = UpwindFormNonDivergence(fes, -(1-2*u)*grad(phi), v, w, h, n)

aupw = BilinearForm(fes)
beta = grad(phi)-grad(g)
etaf = abs(beta*n)
flux = 0.5*(v*(1-v) + v.Other(0)*(1-v.Other(0)))*beta*n
flux += 0.5*etaf*(v-v.Other(0))

#phiR = IfPos(beta*n, beta*n*IfPos(v-0.5, 0.25, v*(1-v)), 0)

aupw += SymbolicBFI(-v*(1-v)*beta*grad(w).Compile())  
aupw += SymbolicBFI(flux*(w - w.Other(0)), VOL, skeleton=True)
#aupw += SymbolicBFI(phiR*w, BND, skeleton=True)

#aupw = UpwindFormDivergence(fes, (1-u)*grad(phi), v, w, h, n)
asip = SIPForm(1, eta, fes, v, w, h, n, Dirichlet=True)
asip.Assemble() # Does not change



rhs = u.vec.CreateVector()
rhs2 = u.vec.CreateVector()
mstar = asip.mat.CreateMatrix()

def step(t, y):
    aupw.Apply (y, rhs)
    # HERE: Replace y by u.vec above works better for some reason even though it should be the same for euler....
#    rhs.data = (-1*tau)*rhs
    asip.Apply(y, rhs2)
    rhs.data += rhs2
#    a.Apply(y, rhs)
    fes.SolveM(rho=CoefficientFunction(1), vec=rhs)
    return -rhs

mstar.AsVector().data = m.mat.AsVector() + tau * D * asip.mat.AsVector()
invmat = mstar.Inverse(fes.FreeDofs())

# Explicit Euler
def HughesSolver(vels):
    t = 0.0
    k = 0
    # Initial data
    xshift = 2
    u.Set(0.8*exp(-((x-xshift)*(x-xshift)+y*y)))
    agents = np.array([0.0]) # Initial pos agents
    phi.Set(5-abs(x))    
    rhodata = []
    phidata = []
    agentsdata = []
    with TaskManager():
        for t in np.nditer(times):
                        
            # Solve Eikonal equation using Newton
            feik.Assemble()
            EikonalSolver()
            
            # FIXME: Only one agent at the moment
            norm = sqrt((x-agents[0])*(x-agents[0])+(y)*(y))
            K = cK*IfPos(1-norm/width, 1-norm/width, 0)
            g.Set(K)
            
            # IMEX Time integration
            aupw.Apply(u.vec,rhs)
            rhs.data = (1*tau)*rhs
            rhs.data += m.mat * u.vec #- tau * a.mat * u.vec
            u.vec.data = invmat * rhs     
            
            # Explicit
#           u.vec.data = RungeKutta(euler, tau, step, t, u.vec)

            # Update Agents positions (expl. Euler)
            agents = agents + tau*vels[:,k]

            rhodata.append(u)
            phidata.append(phi)
            agentsdata.append(agents)
            
            if vtkoutput and k % 10 == 0:
                vtk.Do()                

            
            if netgenMesh.dim == 1:
                stabilityLimiter(u, fes, uplot)
#                nonnegativityLimiter(u, fes, uplot)

            if netgenMesh.dim == 1:
                if k % 100 == 0:
                    uplot.Redraw()
                    phiplot.Redraw()
                    plt.pause(0.001)
            else:
                Redraw(blocking=False)
                
            k += 1
    return [rhodata, phidata, agentsdata]


# Assemble forms for adjoint eq
aupwadj = UpwindFormNonDivergence(fes, (u*(1-u)+u*(1-2*u))*(grad(phi)+grad(g)), v, w, h, n)
## asip stays the same
fadj = LinearForm(fes)
fadj += SymbolicLFI(-(1-2*u)*lam2/(((1-u)*(1-u)+del2)*((1-u)*(1-u)+del2))*w) # FIXME
fadj += SymbolicLFI(gadj*w)

aupwadj2 = UpwindFormNonDivergence(fes, -2*grad(phi), v, w, h, n)
fadj2 = LinearForm(fes)
fadj2 += SymbolicLFI(-u*(1-u)*grad(lam1)*grad(w)) # FIXME


invmat2 = del1*asip.mat.Inverse(fes.FreeDofs())

def AdjointSolver(rhodata, phidata, agentsdata):
    t = 0.0
    k = 0
    # Initial data
    xshift = 2
    lam1.Set(0*x)
    Vs = [] # Save standard deviations to evaluate functional later on
    lam3 = np.zeros(Na)
    with TaskManager():
        for t in np.nditer(times):
            # Read data, backward in time (already reversed)
            u.vec.data = rhodata[k].vec
            phi.vec.data = phidata[k].vec
            agents = agentsdata[k]
            
            norm = sqrt((x-agents[0])*(x-agents[0])+(y)*(y))
            K = cK*IfPos(1-norm/width, 1-norm/width, 0)
            g.Set(K)
            
            E = Integrate(x*u, mesh)
            V = Integrate(u*abs(x-E), mesh)
            Vs.append(V)
            gadj.Set(V*(x-E)*(x-E))
            
            fadj.Assemble() # Assemble RHSs
            fadj2.Assemble()
            
            # IMEX for lam1-Eq
            aupwadj.Apply(u.vec,rhs)
            rhs.data = tau*rhs
            rhs.data += tau*fadj.vec
            rhs.data += m.mat * lam1.vec
            lam1.vec.data = invmat * rhs
            
            # IMEX for lam2-Eq
            aupwadj2.Apply(lam2.vec,rhs)
            rhs.data = tau*rhs
            rhs.data += tau*fadj2.vec
            lam2.vec.data = invmat2 * rhs
            
            # Integrate lam3-equation
            for i in range(0,agents.size):
                norm = sqrt((x-agents[0])*(x-agents[0])+(y)*(y))
                K = cK*IfPos(1-norm/width, 1-norm/width, 0)
                g.Set(K)
                upd = Integrate(u*(1-u)*grad(g)*grad(lam1), mesh)
                lam3[i] = lam3[i] + tau*upd
                vels[i,k] = -Na*tend/alpha*lam3[i]
            
            
            if netgenMesh.dim == 1:
                stabilityLimiter(u, fes, uplot)
                nonnegativityLimiter(u, fes, uplot)
                
            if netgenMesh.dim == 1:
                if k % 1== 0:
                    lam1plot.Redraw()
                    lam2plot.Redraw()
                    plt.pause(0.001)
            else:
                Redraw(blocking=False)

            k += 1
        return [vels, Vs]

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[u, phi],names=["rho","phi"],filename="vtk/rho",subdivision=1)
    vtk.Do()
    
if netgenMesh.dim == 1:
    plt.subplot(221)
    uplot = Plot(u, mesh=mesh)
    plt.title('u')
    plt.subplot(222)
    phiplot = Plot(phi, mesh=mesh)
    plt.title('phi')
    
    plt.subplot(223)
    lam1plot = Plot(lam1, mesh=mesh)
    plt.title('lam1')
    plt.subplot(224)
    lam2plot = Plot(lam2, mesh=mesh)
    plt.title('lam2')
    
    plt.show(block=False)
else:
    Draw(phi, mesh, 'phi')
    Draw(u, mesh, 'u')
    Draw(lam1, mesh, 'lam1')
    Draw(lam2, mesh, 'lam2')

# Gradient descent
Nopt = 10
otau = 1
for k in range(0,Nopt):

    # Solve forward problem
    [rhodata, phidata, agentsdata] = HughesSolver(vels)
    
    # Solve backward problem (call with data already reversed in time)
    [nvels,Vs] = AdjointSolver(rhodata[::-1], phidata[::-1], agentsdata[::-1])
    
    # Update velocities 
    # TODO: Projection auf [-minvel, maxvel]
    vels = vels - otau*nvels
    
    # Evaluate Functional
    J = tau/tend*sum(np.multiply(Vs,Vs))  # 1/T int_0^T |Vs|^2
    for i in range(0,Na):
        J += alpha/(2*Na*tend)*tau*sum(np.multiply(vels[i,:],vels[i,:]))
    
    print('Functional J = ' + str(J))
    print(agentsdata)

    
    
#feik.Assemble()
#EikonalSolver()
#[rhodata, phidata, agentsdata] = HughesSolver(vels)
##
#pickler = NgsPickler(open ("rhodata.dat", "wb"))
#pickler.dump (rhodata)
#del pickler
#pickler = NgsPickler(open ("phidata.dat", "wb"))
#pickler.dump (phidata)
#del pickler
#pickler = NgsPickler(open ("agentsdata.dat", "wb"))
#pickler.dump (agentsdata)
#del pickler

#
#input("Press any key...")
#unpickler = NgsUnpickler(open ("rhodata.dat", "rb"))
#rhodata = unpickler.load()
#del unpickler
#unpickler = NgsUnpickler(open ("phidata.dat", "rb"))
#phidata = unpickler.load()
#del unpickler
#unpickler = NgsUnpickler(open ("agentsdata.dat", "rb"))
#agentsdata = unpickler.load()
#del unpickler
#
