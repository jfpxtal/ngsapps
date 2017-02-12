# Solve the equation
# \rho_t = \nabla \cdot ( D*M(K\ast\rho) \rho)
# with homogeneous Neumman bcs and with
# K(x) = exp(-x) and diff. coeff. D

from netgen.geom2d import unit_square
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *
from netgen.csg import *
import numpy as np
from netgen.geom2d import SplineGeometry

order = 2
conv_order = 3
maxh = 0.09

# time step and end
tau = 0.0001
tend = 25

ngsglobals.msg_level = 1

# diffusion coefficient
D = 1

vtkoutput = False

geo = SplineGeometry()
xmin = -2
xmax = 2
ymin = -1
ymax = 2
dx = xmax - xmin
dy = ymax - ymin
pnts = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
lines = [ (0,1,1,"bottom"), (1,2,2,"right"), (2,3,3,"top"), (3,0,4,"left") ]
pnums = [geo.AppendPoint(*p) for p in pnts]

lbot = geo.Append ( ["line", pnums[0], pnums[1]], bc="bottom")
lright = geo.Append ( ["line", pnums[1], pnums[2]], bc="right")
geo.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, bc="left", copy=lright)
geo.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, bc="top", copy=lbot)

def sqr(x):
    return x*x

# Convolution kernel
thin = 10
k0 = 1
K = k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
# K = IfPos(0.1-sqrt(sqr(x-xPar)+sqr(y-yPar)), 0.1-sqrt(sqr(x-xPar)+sqr(y-yPar)), 0) # k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
# K = CompactlySupportedKernel(radius=0.1, scale=1.0)

#mesh = Mesh(mesh)

#mesh = Mesh (unit_square.GenerateMesh(maxh=0.1))

#geo = SplineGeometry()
#geo.AddCircle ( (0.0, 0.0), r=5, bc="cyl")
#geo.AddRectangle((0, 0), (2, 1), leftdomain=1)

#geo = CSGeometry()
#geo.Add(Torus(Pnt(0,0,0),Vec(-1,0,0),0,1))

mesh = Mesh(geo.GenerateMesh(maxh=maxh))
#urgh
# H1-conforming finite element space
fes = Periodic(H1(mesh, order=order)) # Neumann only, dirichlet=[1,2,3,4])
#fes = L2(mesh, order=4, flags={"dgjumps":True})

u, w = fes.TrialFunction(), fes.TestFunction()

# initial values
s = GridFunction(fes)
#s.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)))
#s.Set(IfPos(RandomCF(0.0,1.0)+1,RandomCF(0.0,1.0)+1,0))
b0 = 4
b1 = 7
sig = 25
s.Set(b0*exp(-sig*(sqr(x-0.5)+sqr(y-0.5)))+b1*exp(-sig*(sqr(x+0.5)+sqr(y-0.5))))

v = GridFunction (fes)

conv = ParameterLF(w*K, s, conv_order, repeat=1, patchSize=[dx, dy])

# the bilinear-form
g = GridFunction(fes)
a = BilinearForm (fes, symmetric=False)
a += SymbolicBFI ( D*(-exp(-g)*u*grad(g) + exp(-g)*grad(u) )*grad(w) )
#a += SymbolicBFI ( D*grad(u)*grad(w) ) # TEST: HEAT EQ
#a += SymbolicBFI ( D*grad(exp(-g)*u)*grad(w) )

m = BilinearForm(fes)
m += SymbolicBFI( u*w )

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
Draw(g, mesh, 'conv')
Draw(s, mesh, 'rho')

if vtkoutput:
    vtk = MyVTKOutput(ma=mesh,coefs=[s],names=["rho"],filename="nonlocaldiffusion",subdivision=3)
    vtk.Do()
#g.Set(1*x) #    conv)
input("")
t = 0.0
with TaskManager():
    while t < tend:
        print("do convolution")
        g.Set(conv)
        print("...done\n")
        a.Assemble()
        smat.AsVector().data = mmat.AsVector() #+ tau * a.mat.AsVector()
        rhs.data = mmat * s.vec - tau * a.mat * s.vec #+ tau*f.vec
        s.vec.data = smat.Inverse(fes.FreeDofs()) * rhs

        t += tau
        print("\n mass = {:10.6e}".format(Integrate(s,mesh)) +  "t = {:10.6e}".format(t))
        Redraw(blocking=False)
        # input("")

        if vtkoutput:
            vtk.Do()
