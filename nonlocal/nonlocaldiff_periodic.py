# Solve the equation
# \rho_t = \nabla \cdot ( M(K\ast\rho) \rho)
# with homogeneous Neumman bcs and with
# K(x) = exp(-x) 

from netgen.geom2d import unit_square
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *
from netgen.csg import *
import numpy as np
from netgen.geom2d import SplineGeometry

def sqr(x):
    return x*x

# FEM parameters
order = 1
conv_order = 2
maxh = 0.17

# time step and final time
tau = 0.1 # Works for all expl w/ sin perturbation
tend = 2500

ngsglobals.msg_level = 1

vtkoutput = False

geo = SplineGeometry()
dsize = 5
xmin, xmax, ymin, ymax = -dsize, dsize, -dsize, dsize
dx = xmax - xmin
dy = ymax - ymin
pnts = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
lines = [ (0,1,1,"bottom"), (1,2,2,"right"), (2,3,3,"top"), (3,0,4,"left") ]
pnums = [geo.AppendPoint(*p) for p in pnts]

lbot = geo.Append ( ["line", pnums[0], pnums[1]], bc="bottom")
lright = geo.Append ( ["line", pnums[1], pnums[2]], bc="right")
geo.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, bc="left", copy=lright)
geo.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, bc="top", copy=lbot)

ngmesh = geo.GenerateMesh()
mesh = Mesh(ngmesh)
ngmesh.Refine()
mesh = Mesh(ngmesh)
ngmesh.Refine()
mesh = Mesh(ngmesh)
ngmesh.Refine()
mesh = Mesh(ngmesh)
ngmesh.Refine()
mesh = Mesh(ngmesh)
ngmesh.Refine()
mesh = Mesh(ngmesh)

# Convolution kernel
thin = 10
k0 = 1
Kmax = 0.6
#K = k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
K2 = k0*IfPos(1-(1/Kmax)*sqrt(sqr(x)+sqr(y)), 1-(1/Kmax)*sqrt(sqr(x)+sqr(y)), 0) # k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
Kint = Integrate(K2,mesh)
K = (1/Kint)*k0*IfPos(1-(1/Kmax)*sqrt(sqr(x-xPar)+sqr(y-yPar)), 1-(1/Kmax)*sqrt(sqr(x-xPar)+sqr(y-yPar)), 0) # k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
#K = CompactlySupportedKernel(radius=0.1, scale=1.0)


# H1-conforming finite element space
fes = Periodic(H1(mesh, order=order)) # Neumann only, dirichlet=[1,2,3,4])
u, w = fes.TrialFunction(), fes.TestFunction()

v = GridFunction (fes)
g = GridFunction(fes)
s = GridFunction(fes)
q = s.vec.CreateVector()
s1 = s.vec.CreateVector()
rhs = s.vec.CreateVector()
conv = ParameterLF(w*K, s, conv_order, repeat=1, patchSize=[dx, dy])

# bilinear-forms
a1 = BilinearForm (fes, symmetric=False)
a2 = BilinearForm (fes, symmetric=False)

a1 += SymbolicBFI ( (0*-exp(-g)*u*grad(g) +   exp(-g)*grad(u) )*grad(w) ) # Implicit
a2 += SymbolicBFI ( (  -exp(-g)*u*grad(g) +  0*exp(-g)*grad(u) )*grad(w) ) # Explicit

m = BilinearForm(fes)
m += SymbolicBFI( u*w )

m.Assemble()
mmat = m.mat
smat = mmat.CreateMatrix()

# initial values
b0, b1 = 4, 7
sig = 25
#s.Set(b0*exp(-sig*(sqr(x-0.5)+sqr(y-0.5)))+b1*exp(-sig*(sqr(x+0.5)+sqr(y-0.5))))
#s.Set(IfPos(RandomCF(0.0,1.0)+1,RandomCF(0.0,1.0)+1,0))
s.Set(1 + 0.17*sin(x)*sin(y)) #RandomCF(0.0,1.0))

# Visualization
g.Set(conv)
Draw(g, mesh, 'K*rho') # K \ast \rho
Draw(s, mesh, 'rho') # \rho


if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[s],names=["rho"],filename="nonlocaldiffusion",subdivision=3)
    vtk.Do()

input("")
t = 0.0
k = 1
with TaskManager():
    while t < tend:
#        print("do convolution")
        g.Set(conv)
 #       print("...done\n")
        
        a1.Assemble()
        a2.Assemble()

   	# Solve using explicit euler
#        a2.Apply (s.vec, q)
#        fes.SolveM (rho=CoefficientFunction(1), vec=q)
	
#        s1.data = s.vec - tau * (mmat.Inverse(fes.FreeDofs())*q) # Two stage SSP-RK

 #       a2.Apply (s1, q)
#        fes.SolveM (rho=CoefficientFunction(1), vec=q)
       
  #      s.vec.data = 0.5*(s.vec) + 0.5*(s1-tau*(mmat.Inverse(fes.FreeDofs())*q) ) # Two stage SSP-RK        
#        


        smat.AsVector().data = mmat.AsVector() + tau * a1.mat.AsVector()
        rhs.data = mmat * s.vec - tau * a2.mat * s.vec
        s.vec.data = smat.Inverse(fes.FreeDofs()) * rhs

        t += tau
        k += 1
        if k % 50 == 0:
            Redraw(blocking=False)	
            print("\n mass = {:10.6e}".format(Integrate(s,mesh)) +  "t = {:10.6e}".format(t))
  

        if vtkoutput:
            vtk.Do()
