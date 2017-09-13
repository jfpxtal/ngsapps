from netgen.geom2d import SplineGeometry
from ngsolve import *

from ngsapps.utils import *
from ngsapps.plotting import *

import matplotlib.pyplot as plt

import geometries as geos
#from stationary import *
from dgform import DGFormulation
from cgform import CGFormulation

class CrossDiffParams:
    def __init__(self, s=None, Dr=None, Db=None, Vr=None, Vb=None):
        self.s = s
        self.Dr = Dr
        self.Db = Db
        self.Vr = Vr
        self.Vb = Vb
#
#def CreateBilinearForm(fes, s, gridr,gridb):
#    r, b = fes.TrialFunction()
#    tr, tb = fes.TestFunction()
#    r2 = s.components[0]
#    b2 = s.components[1]
#    Dr = 0.1
#    Db = 0.1
#
#    a = BilinearForm(fes, symmetric=False)
#    a += SymbolicBFI(Dr*((1-r2-b2)*(grad(gridr)*r+gridr*grad(r))*grad(tr) + r2*gridr*(grad(r)+grad(b))*grad(tr)))
#    a += SymbolicBFI(Db*((1-r2-b2)*(grad(gridb)*b+gridb*grad(b))*grad(tb) + b2*gridb*(grad(r)+grad(b))*grad(tb)))
##    a += SymbolicBFI(0.1*(grad(r)*grad(tr)+grad(b)*grad(tb))) # Regularization
#    return a

order = 1
maxh = 0.05

conv_order = 3

# time step and end
tau = 0.01
tend = -1
conv = True

p = CrossDiffParams()

geo = SplineGeometry()
xmin, xmax = -1, 1
ymin, ymax = -1, 1
dx = xmax-xmin
dy = ymax-ymin
MakePeriodicRectangle(geo, (xmin, ymin), (xmax, ymax))
netmesh = geo.GenerateMesh(maxh=maxh)
mesh = Mesh(netmesh)

fes1 = Periodic(H1(mesh, order=order)) #, flags={'definedon': ['top']}))
fes = FESpace([fes1, fes1])

r, b = fes.TrialFunction()
tr, tb = fes.TestFunction()

# initial values
p.s = GridFunction(fes)
r2 = p.s.components[0]
b2 = p.s.components[1]
# r2.Set(IfPos(0.2-x, IfPos(0.5-y, 0.9, 0), 0))
# b2.Set(IfPos(x-1.8, 0.6, 0))
#r2.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)))
#b2.Set(0.5*exp(-pow(x-1.9, 2)-0.1*pow(y-0.5, 2)))
#freq = 10
#r2.Set(2.0/3*0.5*(sin(freq*x)*sin(freq*y)+1))
#b2.Set(1.0/3*0.5*(cos(freq*x)*cos(freq*y)+1))

r2.Set(RandomCF(0, 0.49))
b2.Set(RandomCF(0, 0.49))
#r2.Set(0.5+0*x)
#b2.Set(0.5+0*x)
#cdec = 10
#cdec2 = 5
#Dr = 1.0/100
#Db = 1.0/100


if conv:
    # convolution
    thin = 30
    k0 = 1
    mK = Integrate(k0*exp(-thin*sqrt(sqr(x)+sqr(y))), mesh)
    k0 = k0/mK
    K = k0*exp(-thin*sqrt(sqr(x-xPar)+sqr(y-yPar))) #/mK
    #K = CompactlySupportedKernel(0.05)

    #K = exp(-sqrt(sqr(x-xPar)*x+sqr(y-yPar)))
    convr = ParameterLF(fes1.TestFunction()*K, r2, conv_order)# repeat=0, patchSize=[dx, dy])
    convb = ParameterLF(fes1.TestFunction()*K, b2, conv_order) #, repeat=0, patchSize=[dx, dy])
else:
    convr = 0
    convb = 0
    
grid = GridFunction(fes)
gridr = grid.components[0]
gridb = grid.components[1]

tmp = GridFunction(fes)
Dr = tmp.components[0]
Db = tmp.components[1]

with TaskManager():
    gridr.Set(convr)
    gridb.Set(k0*exp(-thin*sqrt(sqr(x)+sqr(y))))
    
Draw(r2, mesh, 'r')

Draw(gridr, mesh, 'G*r')
Draw(gridb, mesh, 'G')
