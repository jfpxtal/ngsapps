from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *
from netgen.csg import *
import numpy as np
from netgen.geom2d import SplineGeometry

order = 2
conv_order = 3
maxh = 0.09
# maxh = 0.15
# maxh = 0.3

ngsglobals.msg_level = 4

def sqr(x):
    return x*x

def posPart(x):
    return IfPos(x, x, 0)

def min(x, y):
    return IfPos(x-y, y, x)

def norm(x, y):
    return sqrt(sqr(x)+sqr(y))


geo = SplineGeometry()
xmin = -2
xmax = 2
ymin = -1
ymax = 2
dx = xmax-xmin
dy = ymax-ymin
pnts = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
lines = [ (0,1,1,"bottom"), (1,2,2,"right"), (2,3,3,"top"), (3,0,4,"left") ]
pnums = [geo.AppendPoint(*p) for p in pnts]

lbot = geo.Append ( ["line", pnums[0], pnums[1]], bc="bottom")
lright = geo.Append ( ["line", pnums[1], pnums[2]], bc="right")
geo.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, bc="left", copy=lright)
geo.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, bc="top", copy=lbot)

# pyK = CoefficientFunction(0.0)
# xcenter, ycenter = -1.9, 1.5
# radius = 0.3
# scale = 1.0
# for i in range(-1, 2):
#     for j in range(-1, 2):
#         pyK += scale * posPart(1 - norm(x-xcenter+i*dx, y-ycenter+j*dy)/radius)


mesh = Mesh(geo.GenerateMesh(maxh=maxh))
# K = CompactlySupportedKernel(1)
# K = IfPos(1-sqrt(pow(x-xPar, 2)+sqr(y-yPar)), 1-sqrt(sqr(x-xPar)+sqr(y-yPar)), 0) # k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
# K = IfPos(1-sqrt(sqr(x-xPar)+sqr(y-yPar)), 1-sqrt(sqr(x-xPar)+sqr(y-yPar)), 0) # k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
thin = 1
k0 = 1
K = k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))

# H1-conforming finite element space
fes = Periodic(H1(mesh, order=order)) # Neumann only, dirichlet=[1,2,3,4])
u, w = fes.TrialFunction(), fes.TestFunction()
g = GridFunction(fes)
s = GridFunction(fes)
s.Set(CoefficientFunction(1.0))

conv = ParameterLF(w*K, s, conv_order, repeat=1, patchSize=[dx, dy])

with TaskManager():
    print('conv')
    g.Set(conv)
    print('conv end')
    Draw(g, mesh, 'conv')
    Draw(s, mesh, 's')
    xs = np.linspace(xmin, xmax, 10)
    ys = np.linspace(ymin, ymax, 10)
    input()
    for xx in xs:
        for yy in ys:
            s.Set(posPart(0.2-norm(0.3*(x-xx), y-yy)))
            # s.Set(CoefficientFunction(1.0))
            # xPar.Set(xx)
            # yPar.Set(yy)
            g.Set(conv)
            Redraw()
            input()
