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
from ngsapps.limiter import *
from DGForms import *
from rungekutta import *

import pickle

def f(u):
    return 1-u

def fprime(u):
    return -1 + 0*u


ngsglobals.msg_level = 0
plotmod = 1000

order = 1
tau = 0.01
tend = 3 # 3.5
vtkoutput = False

# Regularization parameters
del1 = 0.1
del2 = 0.1
D = 0.05
alpha = 0.01

cK = 1 # Parameter to control strength of attraction
sigK = 2 # Sigma of exponential conv kernel
width = 1 # width of conv kernel

# Gradient descent
otau = 0.6

eta = 5 # Penalty parameter


usegeo = "circle"
# usegeo = "1d"

if usegeo == "circle":
    radius = 4
    maxh = 0.15
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=radius, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)

    u_init = 0.8*exp(-(sqr(x+1)+0.2*sqr(y)))
    # u_init += 0.8*exp(-(sqr(x+1)+sqr(y-1)))
    # u_init += 0.8*exp(-(sqr(x)+sqr(y)))
    phi_init = radius-norm(x, y)
    ag_init = [(1,1),(1,-1)]

elif usegeo == "1d":
    radius = 8
    maxh = 0.1
    netgenMesh = Make1DMesh(-radius, radius, maxh)

    # initial data
    u_init = 0.8*exp(-(sqr(x-3)))+0.8*exp(-(sqr(x-6)))
    phi_init = radius-abs(x)
    ag_init = [(3,)]

mesh = Mesh(netgenMesh)

Na = len(ag_init) # Number of agents

times = np.linspace(0.0,tend,np.ceil(tend/tau)) # FIXME: make tend/tau integer
# Inital velocity of agents
vels = np.zeros((times.size, Na, mesh.dim))
# vels = 0.8*np.ones((times.size, Na, mesh.dim))
# vels = -0.5*np.ones((times.size, Na, mesh.dim))

def K(agent):
    if mesh.dim == 1:
        return cK*exp(-sigK*(sqr(x-agent[0])))
    else:
        return cK*exp(-sigK*(sqr(x-agent[0])+sqr(y-agent[1])))

def Kprime(agent):
    if mesh.dim == 1:
        return [2*sigK*K(agent)*(x-agent[0])]
    else:
        return 2*sigK*K(agent)*CoefficientFunction((x-agent[0], y-agent[1]))

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

q = u.vec.CreateVector()
q2 = u.vec.CreateVector()

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(v*w)
print('Assembling m...')
m.Assemble()
minv = m.mat.Inverse(fes.FreeDofs())

asip = SIPForm(1, eta, fes, v, w, h, n, Dirichlet=True)
asip.Assemble()

# Eikonal equation forms
# Upwind for Hamilton Jacobi term:
aeupw = UpwindFormNonDivergence(fes, grad(phi), v, w, h, n, Compile=False)

feik = LinearForm(fes)
feik += SymbolicLFI(1/(sqr(f(u))+del2)*w)

mstar = m.mat.CreateMatrix()

# Solve reg. Eikonal equation usign Newton's scheme
def EikonalSolver():
    k = 0
    errNewton = 1e99
    while errNewton > 1e-6:
        k += 1

        # Apply nonlinear operator
        aeupw.Assemble()
        q.data = aeupw.mat*phi.vec
        q.data += del1*asip.mat*phi.vec
        q.data -= feik.vec

        # Linearized HJ-Operator - has additional 2 from square
        mstar.AsVector().data = del1*asip.mat.AsVector() + 2*aeupw.mat.AsVector()

        # Solve for update and perform Newton step
        invmat = mstar.Inverse(fes.FreeDofs())
        phi.vec.data -= invmat * q

        errNewton = q.Norm()
    #print('Newton finished with Res error = ' + str(q.Norm()) + ' after ' + str(k) + 'steps \n') # L2norm of update

# Forms for Hughes Model diff-transport eq
eiksolver = EikonalSolver2D(fes, [(-radius,-radius), (radius,-radius), (radius,radius), (-radius,radius)])
phi = eiksolver.GetSolutionGF()

aupw = BilinearForm(fes)
beta = grad(phi)-grad(g)
etaf = abs(beta*n)
flux = 0.5*(v*sqr(f(v))*beta + v.Other()*sqr(f(v.Other()))*beta.Other())*n
flux += 0.5*etaf*(v-v.Other(0))

aupw += SymbolicBFI(-v*sqr(f(v))*beta*grad(w))
aupw += SymbolicBFI(flux*(w - w.Other()), VOL, skeleton=True)

rhs = u.vec.CreateVector()
rhs2 = u.vec.CreateVector()

mstar.AsVector().data = m.mat.AsVector() + tau * D * asip.mat.AsVector()
invmat = mstar.Inverse(fes.FreeDofs())

# Explicit Euler
def HughesSolver(vels):
    # Initial data
    u.Set(u_init)
    agents = ag_init[:]
    # phi.Set(phi_init)
    rhodata = np.empty((len(times), u.vec.size))
    phidata = np.empty((len(times), phi.vec.size))
    agentsdata = np.empty_like(vels)

    for k, t in enumerate(times):
        # # Solve Eikonal equation using Newton
        # feik.Assemble()
        # EikonalSolver()

        # UnregEikonal1D()
        # SolveEikonal1D(1/sqrt(sqr(f(u))+del2), phi)
        # SolveEikonal1D(1/f(u), phi)
        eiksolver.Solve(1/f(u))

        gcf = 0
        for ag in agents:
            gcf += K(ag)
        g.Set(gcf/Na)

        # IMEX Time integration
        aupw.Apply(u.vec, rhs)
        rhs.data = tau*rhs
        rhs.data += m.mat * u.vec
        u.vec.data = invmat*rhs

        # Update Agents positions (expl. Euler)
        agents += tau*vels[k,:,:]

        rhodata[k,:] = u.vec.FV()
        phidata[k,:] = phi.vec.FV()
        agentsdata[k,:,:] = agents

        if vtkoutput and k % 10 == 0:
            vtk.Do()

        Limit(u, p1fes, 1, 1, maxh, True)

        if mesh.dim == 1:
            if (k+1) % plotmod == 0 or t == times[-1]:
                uplot.Redraw()
                phiplot.Redraw()
                gplot.Redraw()
                plt.pause(0.001)
                print('Hughes @ t = ' + str(t) + ', mass = ' + str(Integrate(u,mesh)) + '\n')
        else:
            Redraw(blocking=False)

    return [rhodata, phidata, agentsdata]


# Assemble forms for adjoint eq
aupwadj = UpwindFormNonDivergence(fes, (f(u)*f(u) + 2*u*f(u)*fprime(u))*(grad(phi)-grad(g)), v, w, h, n)
## asip stays the same
fadj = LinearForm(fes)
fadj += SymbolicLFI(-2*fprime(u)*f(u)*lam2/(sqr(sqr(f(u))+del2))*w)
fadj += SymbolicLFI(gadj*w)

aupwadj2 = BilinearForm(fes)
beta = 2*grad(phi)
etaf = abs(beta*n)
flux = 0.5*(v*beta + v.Other()*beta.Other())*n
flux += 0.5*etaf*(v-v.Other())

aupwadj2 += SymbolicBFI(-v*beta*grad(w))
aupwadj2 += SymbolicBFI(flux*(w - w.Other(0)), VOL, skeleton=True)

fadj2 = LinearForm(fes)
fadj2 += SymbolicLFI(-u*f(u)*f(u)*grad(lam1)*grad(w))

invmat2 = del1*asip.mat.Inverse(fes.FreeDofs())

def AdjointSolver(rhodata, phidata, agentsdata, vels):
    # Initial data
    lam1.Set(0*x)
    Vs = np.zeros_like(times) # Save variances to evaluate functional later on
    lam3 = np.zeros((Na, mesh.dim))
    nvels = alpha/(Na*tend)*vels # Local vels

    for k in range(len(times)):
        # Read data, backward in time (already reversed)
        u.vec.FV().NumPy()[:] = rhodata[k,:]
        phi.vec.FV().NumPy()[:] = phidata[k,:]
        agents = agentsdata[k,:,:]

        gcf = 0
        for ag in agents:
            gcf += K(ag)
        g.Set(gcf/Na)

        mass = Integrate(u, mesh)
        if mesh.dim == 1:
            E = Integrate(x*u/mass, mesh)
            V = Integrate(u/mass*sqr(x-E), mesh)
            gadj.Set((1/tend)*V*sqr(x-E)/mass) # mass deriv?
        else:
            E = Integrate(CoefficientFunction((x,y))*u/mass, mesh)
            V = Integrate(u/mass*(sqr(x-E[0])+sqr(y-E[1])), mesh)
            gadj.Set((1/tend)*V*(sqr(x-E[0])+sqr(y-E[1]))/mass) # mass deriv?

        Vs[k] = V

        fadj.Assemble()
        fadj2.Assemble()

        # IMEX for lam1-Eq
        aupwadj.Apply(lam1.vec,rhs)
        rhs.data = -tau*rhs
        rhs.data += tau*fadj.vec
        rhs.data += m.mat * lam1.vec
        lam1.vec.data = invmat * rhs

        Limit(lam1, p1fes, 1, 0.1, maxh, False)

        # IMEX for lam2-Eq
        aupwadj2.Apply(lam2.vec,rhs)
        rhs.data += fadj2.vec
        lam2.vec.data = invmat2 * rhs

        Limit(lam2, p1fes, 1, 0.1, maxh, False)

        # Integrate lam3-equation
        for i in range(Na):
            kprime = Kprime(agents[i])
            for j, kp in enumerate(kprime):
                g.Set(kp/Na)
                upd = Integrate(u*sqr(f(u))*grad(g)*grad(lam1), mesh)
                lam3[i,j] += tau*upd
                nvels[k,i,j] += lam3[i,j]

        if mesh.dim == 1:
            if (k+1) % plotmod == 0 or k == len(times)-1:
                lam1plot.Redraw()
                lam2plot.Redraw()
                plt.pause(0.001)
        else:
            Redraw(blocking=False)

    return [nvels[::-1,:,:], Vs[::-1]]

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[u, phi],names=["rho","phi"],filename="vtk/rho",subdivision=1)
    vtk.Do()

if mesh.dim == 1:
    plt.subplot(421)
    uplot = Plot(u, mesh=mesh)
    plt.title('u')
    plt.subplot(422)
    phiplot = Plot(phi, mesh=mesh)
    plt.title('phi')
 
    plt.subplot(423)
    lam1plot = Plot(lam1, mesh=mesh)
    plt.title('lam1')
    plt.subplot(424)
    lam2plot = Plot(lam2, mesh=mesh)
    plt.title('lam2')

    plt.subplot(413)
    gplot = Plot(g, mesh=mesh)
    plt.title('K')

    # Plot agent positions
    plt.subplot(414)
    ax = plt.gca()
    line_x = []
    for i in range(Na):
        line_x.append(ax.plot(times,times)[0])
    plt.title('Position agent')

else:
    Draw(lam1, mesh, 'lam1')
    Draw(lam2, mesh, 'lam2')
    Draw(g, mesh, 'g')
    Draw(phi, mesh, 'phi')
    # Draw(phi2, mesh, 'phi2')
    Draw(u, mesh, 'u')

# Plot variance
plt.figure()
plt.subplot(211)
axv = plt.gca()
linev_x, = axv.plot(times,times)
plt.title('Variance')

# Plot functional
plt.subplot(212)
axJ = plt.gca()
lJ1, = plt.plot([], [], 'r')
lJ2, = plt.plot([], [], 'g')
lJ, = plt.plot([], [], 'b')
plt.title('J')

plt.show(block=False)

# pick = NgsPickler(open('circ_catch2.dat', 'wb'))

def run(vels):
    k = 0
    with TaskManager():
        while True:

            # Solve forward problem
            rhodata, phidata, agentsdata = HughesSolver(vels)

            # Solve backward problem (call with data already reversed in time)
            nvels, Vs = AdjointSolver(rhodata[::-1,:], phidata[::-1,:], agentsdata[::-1,:,:], vels[::-1,:,:])

            # Plot
            if mesh.dim == 1:
                # Update agents plot
                for i in range(Na):
                    line_x[i].set_ydata(agentsdata[:,i,0])
                ax.relim()
                ax.autoscale_view()

            linev_x.set_ydata(Vs)
            axv.relim()
            axv.autoscale_view()

            # Update velocities
            vels -= otau*nvels

            # Project to interval [-radius/tend, radius/tend]
            vels = np.minimum(vels,0.5*radius/tend)
            vels = np.maximum(vels,-0.5*radius/tend)

            # Evaluate Functional
            J1 = tau/(2*tend)*np.vdot(Vs,Vs)  # 1/T int_0^T |Vs|^2
            print('Functional J_1 = ' + str(J1))
            J2 = alpha/(2*Na*tend)*np.vdot(vels, vels)
            J = J1 + J2
            print('Functional J = ' + str(J))

            lJ1.set_xdata(list(range(k+1)))
            lJ2.set_xdata(list(range(k+1)))
            lJ.set_xdata(list(range(k+1)))
            lJ1.set_ydata(np.hstack((lJ1.get_ydata(), J1)))
            lJ2.set_ydata(np.hstack((lJ2.get_ydata(), J2)))
            lJ.set_ydata(np.hstack((lJ.get_ydata(), J)))
            axJ.relim()
            axJ.autoscale_view()
            plt.pause(0.001)

            # pick.dump([rhodata, phidata, agentsdata, nvels, Vs, vels, J1, J2, J])

            k += 1

# import cProfile
# cProfile.run('run(vels)', 'statsnew')
# input('done')
run(vels)
