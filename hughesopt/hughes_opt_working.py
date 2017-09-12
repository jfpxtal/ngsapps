#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:26:14 2017
@author: pietschm
"""
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

def f(u):
    return 1-u

def fprime(u):
    return -1 + 0*u

order = 1
maxh = 0.05
tau = 0.01
tend = 3 # 3.5
times = np.linspace(0.0,tend,np.ceil(tend/tau)) # FIXME: make tend/tau integer
vtkoutput = False

del1 = 0.1 # Regularization parameters
del2 = 0.1
D = 0.05

Na = 1 # Number of agents
width = 1 # width of conv kernel
alpha = 0.01 # Regularization parameter
cK = 1 # Parameter to control strength of attraction
sigK = 2 # Sigma of exponential conv kernel
vels = 0.8*np.ones((Na,times.size)) # Position of agents

eta = 5 # Penalty parameter

usegeo = "circle"
usegeo = "1d"

plotting = True

radius = 8

if usegeo == "circle":
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=radius, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)
elif usegeo == "1d":
    netgenMesh = Make1DMesh(-radius, radius, maxh)

mesh = Mesh(netgenMesh)

# finite element space
fes = L2(mesh, order=order, flags={'dgjumps': True})
p1fes = L2(mesh, order=1, flags={'dgjumps': True})
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
aeupw = UpwindFormNonDivergence(fes, grad(phi_rhs), v, w, h, n, Compile=False)# Updwind for Hamilton Jacobi term
feik = LinearForm(fes)
feik += SymbolicLFI(1/(sqr(f(u))+del2)*w) # FIXME

feik.Assemble()
a.Assemble()
mstar = m.mat.CreateMatrix()
a.Apply(phi.vec, q)

rhofinal = np.zeros([times.size, u.vec.size])

# Solve reg. Eikonal equation usign Newton's scheme
def EikonalSolver():
    tau = 1
    k = 0
    errNewton = 1e99
    while errNewton > 1e-6:
        k += 1

        # Apply nonlinear operator
        phi_rhs.vec.data = phi.vec
        aeupw.Assemble()
        q.data = aeupw.mat*phi.vec
        q.data += a.mat*phi.vec
        q.data -= feik.vec

        # Linearized HJ-Operator - has additional 2 from square
        mstar.AsVector().data = a.mat.AsVector() + 2*aeupw.mat.AsVector()

        # Solve for update and perform Newton step
        invmat = mstar.Inverse(fes.FreeDofs())
        phi.vec.data -= invmat * q

        errNewton = q.Norm()
    #print('Newton finished with Res error = ' + str(q.Norm()) + ' after ' + str(k) + 'steps \n') # L2norm of update

# Forms for Hughes Model diff-transport eq
#aupw = UpwindFormNonDivergence(fes, -(1-2*u)*grad(phi), v, w, h, n)

aupw = BilinearForm(fes)
beta = grad(phi)-grad(g)
#beta = -grad(g)
etaf = abs(beta*n)
flux = 0.5*(v*f(v)*f(v) + v.Other(0)*f(v.Other(0))*f(v.Other(0)))*beta*n
flux += 0.5*etaf*(v-v.Other(0))

#phiR = IfPos(beta*n, beta*n*IfPos(v-0.5, 0.25, v*(1-v)), 0)

aupw += SymbolicBFI(-v*f(v)*f(v)*beta*grad(w))
aupw += SymbolicBFI(flux*(w - w.Other(0)), VOL, skeleton=True)
#aupw += SymbolicBFI(phiR*w, BND, skeleton=True)

#aupw = UpwindFormDivergence(fes, (1-u)*grad(phi), v, w, h, n)
asip = SIPForm(1, eta, fes, v, w, h, n, Dirichlet=True)
asip.Assemble() # Does not change

rhs = u.vec.CreateVector()
rhs2 = u.vec.CreateVector()
mstar = asip.mat.CreateMatrix()


mstar.AsVector().data = m.mat.AsVector() + tau * D * asip.mat.AsVector()
invmat = mstar.Inverse(fes.FreeDofs())

# Explicit Euler
def HughesSolver(vels, control=True):
    t = 0.0
    k = 0
    # Initial data
    mi = 1# Integrate(unitial, mesh)
    u.Set(1/mi*unitial)
    agents = np.array([1.0]) # Initial pos agents
    phi.Set(5-abs(x))
    rhodata = np.zeros([times.size, u.vec.size])
    phidata = []
    agentsdata = []

    for t in np.nditer(times):
        # Solve Eikonal equation using Newton
        feik.Assemble()
        EikonalSolver()

        # FIXME: Only one agent at the moment
        if control:
          norm = sqrt(sqr(x-agents[0])+y*y)
          K = cK*exp(-sigK*sqr(norm)) # posPart(1-norm/width)
          #g.Set(-2*(x-agents[0])*sigK*K)
          g.Set(K)
        else:
          g.Set(0*x)

        # IMEX Time integration
        aupw.Apply(u.vec,rhs)
        rhs.data = (1*tau)*rhs
        rhs.data += m.mat * u.vec #- tau * a.mat * u.vec
        u.vec.data = invmat * rhs

        # Explicit
#           u.vec.data = RungeKutta(euler, tau, step, t, u.vec)

        # Update Agents positions (expl. Euler)
        agents = agents + tau*vels[:,k] # FIXME

        #rhodata.append(u.vec.FV()[:])
        rhodata[k,:]= u.vec.FV().NumPy()[:]
        phidata.append(phi.vec.FV()[:])
        agentsdata.append(agents)

        if vtkoutput and k % 10 == 0:
            vtk.Do()


        if netgenMesh.dim == 1:
            stabilityLimiter(u, p1fes)
            nonnegativityLimiter(u, p1fes)

        if netgenMesh.dim == 1:
            if k % 50 == 0 and plotting:
                uplot.Redraw()
                phiplot.Redraw()
                gplot.Redraw()
                plt.pause(0.001)
                #print('Hughes @ t = ' + str(t) + ', mass = ' + str(Integrate(u,mesh)))
        else:
            Redraw(blocking=False)

        k += 1

    return [rhodata, phidata, agentsdata]


# Assemble forms for adjoint eq
aupwadj = UpwindFormNonDivergence(fes, (f(u)*f(u) + 2*u*f(u)*fprime(u))*(grad(phi)-g), v, w, h, n)
## asip stays the same
fadj = LinearForm(fes)
fadj += SymbolicLFI(-2*fprime(u)*f(u)*lam2/(sqr(sqr(f(u))+del2))*w) # FIXME
fadj += SymbolicLFI(gadj*w)

#aupwadj2 = UpwindFormNonDivergence(fes, -2*grad(phi), v, w, h, n)
aupwadj2 = BilinearForm(fes)
beta = 2*grad(phi)
etaf = abs(beta*n)
flux = 0.5*(v*beta + v.Other(0)*beta.Other())*n
flux += 0.5*etaf*(v-v.Other(0))

#phiR = IfPos(beta*n, beta*n*IfPos(v-0.5, 0.25, v*(1-v)), 0)

aupwadj2 += SymbolicBFI(-v*beta*grad(w))
aupwadj2 += SymbolicBFI(flux*(w - w.Other(0)), VOL, skeleton=True)

fadj2 = LinearForm(fes)
fadj2 += SymbolicLFI(-u*f(u)*f(u)*grad(lam1)*grad(w)) # FIXME

invmat2 = del1*asip.mat.Inverse(fes.FreeDofs())

def AdjointSolver(rhodata, phidata, agentsdata, vels):
    t = 0.0
    k = 0
    # Initial data
    lam1.Set(0*x)
    Vs = np.zeros(times.size) # Save standard deviations to evaluate functional later on
    lam3 = np.zeros(Na)
    nvels = alpha/(Na*tend)*vels # Local vels
    for t in np.nditer(times):
        # Read data, backward in time (already reversed)
        u.vec.FV().NumPy()[:] = rhodata[k]
        phi.vec.FV()[:] = phidata[k]
        agents = agentsdata[k]

        norm = sqrt(sqr(x-agents[0])+y*y)
        K = cK*exp(-sigK*sqr(norm)) # posPart(1-norm/width)
        g.Set(-2*(x-agents[0])*sigK*K)

        mass = Integrate(u, mesh)
        E = Integrate(x*u/mass, mesh)
        V = Integrate(u/mass*sqr(x-E), mesh)
        Vs[k] = V
        gadj.Set((1/tend)*V*sqr(x-E)/mass)

        fadj.Assemble() # Assemble RHSs
        fadj2.Assemble()

        # IMEX for lam1-Eq
        aupwadj.Apply(lam1.vec,rhs)
        rhs.data = -tau*rhs
        rhs.data += tau*fadj.vec
        rhs.data += m.mat * lam1.vec
        lam1.vec.data = invmat * rhs

        if netgenMesh.dim == 1:
            stabilityLimiter(lam1, p1fes)

        # IMEX for lam2-Eq
        aupwadj2.Apply(lam2.vec,rhs)
        rhs.data += fadj2.vec
        lam2.vec.data = invmat2 * rhs

        if netgenMesh.dim == 1:
            stabilityLimiter(lam2, p1fes)

        # Integrate lam3-equation
        for i in range(0,agents.size):
            norm = sqrt(sqr(x-agents[0])+y*y)
            K = cK*exp(-sigK*sqr(norm)) # posPart(1-norm/width)
#            g.Set(-2*sigK*K*(1-2*sigK*sqr(x-agents[0]))) # 2nd derivate
            g.Set(-2*(x-agents[0])*sigK*K) # 1st derivative
            upd = (1/Na)*Integrate(u*sqr(f(u))*grad(g)*grad(lam1), mesh)
            lam3[i] = lam3[i] - tau*upd
            nvels[i,k] += lam3[i]

        if netgenMesh.dim == 1:
            if k % 100 == 0 and plotting:
                lam1plot.Redraw()
                lam2plot.Redraw()
                plt.pause(0.001)
        else:
            Redraw(blocking=False)

        k += 1
    return [nvels[:,::-1], Vs]

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[u, phi],names=["rho","phi"],filename="vtk/rho",subdivision=1)
    vtk.Do()

if netgenMesh.dim == 1 and plotting:
    plt.subplot(221)
    uplot = Plot(u, mesh=mesh)
    line_x, = plt.plot([0], [1], marker='o', markersize=3, color="red")
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

    plt.figure()
    plt.subplot(311)
    gplot = Plot(g, mesh=mesh)
    plt.title('K')

    # Plot agents position
    plt.subplot(313)
    ax = plt.gca()
    line_x, = ax.plot(times,times)
    plt.title('Position agent')

    # Plot variance
    plt.subplot(312)
    axv = plt.gca()
    linev_x, = axv.plot(times,times)
    plt.title('Variance')

    plt.show(block=False)

elif plotting:
    Draw(phi, mesh, 'phi')
    Draw(u, mesh, 'u')
    Draw(lam1, mesh, 'lam1')
    Draw(lam2, mesh, 'lam2')

# Gradient descent
Nopt = 20
otau = 0.6

#sad
xshift = 2
unitial = 0.8*exp(-(sqr(x-xshift)+y*y))

Vopt = []
Jopt = []
with TaskManager():
#    for k in range(Nopt):
    graderr = 1e10

    while graderr > 0.05:

        # Solve forward problem
        #        import cProfile
        #        cProfile.run('HughesSolver(vels)')
        [rhodata, phidata, agentsdata] = HughesSolver(vels)
        
#        pickler = pickle.Pickler(open ("rhodata_nocontrol.dat", "wb"))
#        pickler.dump (rhodata)
#        del pickler
        
#        afdlk

    #    from matplotlib.widgets import Slider
    #    fig_sol, (ax, ax_slider) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[10, 1]})

    #    vplot = Plot(u, ax=ax, mesh=mesh)
    #    slider = Slider(ax_slider, "Time", 0, tend)

    #    def update(t):
    #        k = int(t/tau)
    #        u.vec.FV()[:] = rhodata[k]
    #        vplot.Redraw()
    #        #plt.pause(0.000001)
    #
    #    slider.on_changed(update)
    #    plt.show(block=False)

        # Solve backward problem (call with data already reversed in time)
        [nvels,Vs] = AdjointSolver(rhodata[::-1,:], phidata[::-1], agentsdata[::-1], vels[:,::-1])

        Vopt.append(Vs)

        # print(Vs)
        # Plot

        # Update agents plot
        if plotting:
            line_x.set_ydata(agentsdata)
            ax.relim()
            ax.autoscale_view()

            linev_x.set_ydata(Vs)
            axv.relim()
            axv.autoscale_view()
            lam1plot.Redraw()
            lam2plot.Redraw()
            uplot.Redraw()
            phiplot.Redraw()
            plt.pause(0.001)

        # Update velocities
        vels = vels - otau*nvels

        graderr = sqrt(tau*sum(np.multiply(nvels[0,:],nvels[0,:])))

        # Project to interval [-radius/tend, radius/tend]
        #vels = np.minimum(vels,0.5*radius/tend)
        #vels = np.maximum(vels,-0.5*radius/tend)

    #    print(nvels)
        #print(vels)
          #  asd

        # Evaluate Functional
        J = tau/(2*tend)*sum(np.multiply(Vs,Vs))  # 1/T int_0^T |Vs|^2
        #print('Functional J_1 = ' + str(J))
        for i in range(0,Na):
            J += alpha/(2*Na*tend)*tau*sum(np.multiply(vels[i,:],vels[i,:]))

        Jopt.append(J)

        print('Functional J = ' + str(J) + ' Gradient = ' + str(graderr) + '\n')
    #    input("press key")
        #print(agentsdata)

#feik.Assemble()
#EikonalSolver()
#[rhodata, phidata, agentsdata] = HughesSolver(vels)
##
pickler = pickle.Pickler(open ("veldata.dat", "wb"))
pickler.dump (vels)
del pickler

pickler = pickle.Pickler(open ("variancedata.dat", "wb"))
pickler.dump (Vopt)
del pickler

pickler = pickle.Pickler(open ("agentsdata.dat", "wb"))
pickler.dump (agentsdata)
del pickler

pickler = pickle.Pickler(open ("rhodata.dat", "wb"))
pickler.dump (rhodata)
del pickler

pickler = pickle.Pickler(open ("Jdata.dat", "wb"))
pickler.dump (Jopt)
del pickler


# Save uncontrolled system
[rhodata, phidata, agentsdata] = HughesSolver(vels, control=False)

pickler = pickle.Pickler(open ("rhodata_nocontrol.dat", "wb"))
pickler.dump (rhodata)
del pickler


#unpickler = pickle.Unpickler(open ("rhodata.dat", "rb"))
#urh = unpickler.load()
#del unpickler


#pickler = NgsPickler(open ("agentsdata.dat", "wb"))
#pickler.dump (agentsdata)
#del pickler

#

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
