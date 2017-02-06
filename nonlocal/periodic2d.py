from netgen.geom2d import SplineGeometry
from ngsolve import *
from math import pi

# ngsglobals.numthreads = 1


periodic = SplineGeometry()
xmin = 0
xmax = 3
ymin = -1
ymax = 1
pnts = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
lines = [ (0,1,1,"bottom"), (1,2,2,"right"), (2,3,3,"top"), (3,0,4,"left") ]
pnums = [periodic.AppendPoint(*p) for p in pnts]

lbot = periodic.Append ( ["line", pnums[0], pnums[1]], bc="bottom")
lright = periodic.Append ( ["line", pnums[1], pnums[2]], bc="right")
periodic.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, bc="left")
periodic.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, bc="top", copy=lbot)

mesh = periodic.GenerateMesh(maxh=0.2)
mesh = Mesh(mesh)

fes = L2(mesh, order=4, flags={"dgjumps":True})

u = fes.TrialFunction()
v = fes.TestFunction()

hq = 1#6.626e-34
e = 1#1.602e-19
E = CoefficientFunction(0.5)
epsdk = CoefficientFunction((-0.5)*sin(pi*y))
b = CoefficientFunction( (1/hq * epsdk, e*E/hq) )
bn = b*specialcf.normal(2)

ubnd = CoefficientFunction(0)

a = BilinearForm(fes)
a += SymbolicBFI (-u * b*grad(v))
a += SymbolicBFI ( bn*IfPos(bn, u, u.Other()) * (v-v.Other()), VOL, skeleton=True)
a += SymbolicBFI ( bn*IfPos(bn, u, u.Other(bnd=ubnd)) * v, BND, skeleton=True)

u = GridFunction(fes)
pos = (pi/2,0)
u.Set(exp (-50 * ( (x-pos[0])*(x-pos[0]) + (y-pos[1])*(y-pos[1]) )))

w = u.vec.CreateVector()

Draw (u, autoscale = False, sd=2)

t = 0
tau = 1e-3#5e-36
tend = 1e4*tau

# input('start')
import time
t1 = time.time()
with TaskManager():
    while t < tend-tau/2:
        a.Apply (u.vec, w)
        fes.SolveM (rho=CoefficientFunction(1), vec=w)
        u.vec.data -= tau * w
        t += tau
        Redraw()
        print('t = {:.4f}'.format(t))
print("t = ",time.time()-t1)
