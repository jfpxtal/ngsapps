# Solve the equation
# \rho_t = \nabla \cdot ( D*M(K\ast\rho) \rho)
# with homogeneous Neumman bcs and with
# K(x) = exp(-x) and diff. coeff. D

from netgen.geom2d import unit_square
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *
import numpy as np

order = 2
conv_order = 3
maxh = 0.2

# time step and end
tau = 0.01
tend = 25

ngsglobals.msg_level = 1

# diffusion coefficient
D = 1

# Convolution kernel
thin = 1
k0 = 1
K = k0*exp(-thin*(x*x+y*y))

vtkoutput = True

#mesh = Mesh (unit_square.GenerateMesh(maxh=0.1))
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddCircle ( (0.0, 0.0), r=5, bc="cyl")
#geo.AddRectangle((0, 0), (2, 1), leftdomain=1)
mesh = Mesh(geo.GenerateMesh(maxh=maxh))

# H1-conforming finite element space
fes = H1(mesh, order=order) # Neumann only, dirichlet=[1,2,3,4])
u = fes.TrialFunction()
w = fes.TestFunction()

# initial values
s = GridFunction(fes)
#s.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)))
s.Set(IfPos(RandomCF(0.0,1.0)+1,RandomCF(0.0,1.0)+1,0))

v = GridFunction (fes)

conv = Convolve(s, K, mesh, conv_order)

# the bilinear-form
g = GridFunction(fes)
a = BilinearForm (fes, symmetric=False)
a += SymbolicBFI ( D*(-exp(-g)*u*grad(g)*grad(w) + exp(-g)*grad(u)*grad(w) ) )
#a += SymbolicBFI ( D*grad(exp(-g)*u)*grad(w) )
m = BilinearForm(fes)
m += Mass(1)

f = LinearForm(fes)
#f += SymbolicLFI ( alpha1*tr + alpha2*tb,BND,definedon=[1] )
f.Assemble()

m.Assemble()
mmat = m.mat
smat = mmat.CreateMatrix()

# Calculate constant equilibria
rhs = v.vec.CreateVector()
uold = v.vec.CreateVector()

# visualize both species at the same time, red in top rectangle, blue in bottom
# translate density b2 of blue species to bottom rectangle
Draw(s, mesh, 'rho')

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[s],names=["rho"],filename="nonlocaldiffusion",subdivision=3)
    vtk.Do()

input("")
t = 0.0
with TaskManager():
    while t < tend:
        print("do convolution")
        g.Set(conv)
        print("...done\n")
        a.Assemble()
        smat.AsVector().data = tau * a.mat.AsVector() + mmat.AsVector()
        rhs.data = mmat * s.vec + tau*f.vec
        s.vec.data = smat.Inverse(fes.FreeDofs()) * rhs
    
        t += tau
        print("\n mass = {:10.6e}".format(Integrate(s,mesh)) +  "t = {:10.6e}".format(t))
        Redraw(blocking=False)
    
        if vtkoutput:
            vtk.Do()    