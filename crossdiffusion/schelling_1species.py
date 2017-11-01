from netgen.geom2d import SplineGeometry
from ngsolve import *
import numpy as np
import time


from ngsapps.utils import *
from ngsapps.plotting import *

import matplotlib.pyplot as plt

import geometries as geos
#from stationary import *

ngsglobals.msg_level = 0

order = 1
maxh = 0.015

conv_order = 2

# time step and end
tau = 0.001
tend = -1

conv = True
vtkoutput = False

yoffset = -1.3

usegeo = "square"
usegeo = "1d"

if usegeo == "square":
    geo = SplineGeometry()
    xmin, xmax = -1/2, 1/2
    ymin, ymax = -1/2, 1/2
    dx = xmax-xmin
    dy = ymax-ymin
    MakePeriodicRectangle(geo, (xmin, ymin), (xmax, ymax))
    netmesh = geo.GenerateMesh(maxh=maxh)
    
elif usegeo == "1d":
    radius = 1/2
    dx = 2*radius
    dy = 0
    maxh = 0.01
    netmesh = Make1DMesh(-radius, radius, maxh, periodic=True)

mesh = Mesh(netmesh)

fes = Periodic(H1(mesh, order=order)) #, flags={'definedon': ['top']}))

r = fes.TrialFunction()
tr = fes.TestFunction()

# initial values
r2 = GridFunction(fes)
freq = 10

#r2.Set(RandomCF(0, 0.49))
r2.Set(0.3 + 0.02*sin(8*3.14159*x))
cdec = 10
#Dr = 1.0/100
#Db = 1.0/100


if conv:
    # convolution
    thin = 40
    k0 = 1
    mK = Integrate(k0*exp(-thin*sqrt(sqr(x)+sqr(y))), mesh)
    k0 = k0/mK
    K = k0*exp(-thin*sqrt(sqr(x-xPar)+sqr(y-yPar))) #/mK
    print("mK = " + str(mK))
    #K = CompactlySupportedKernel(0.05)

    #K = exp(-sqrt(sqr(x-xPar)*x+sqr(y-yPar)))
    conv = ParameterLF(fes.TestFunction()*K, r2, conv_order, repeat=1, patchSize=[dx, dy])
else:
    asd
    conv = 0
    

kernel = GridFunction(fes)
kernel.Set(k0*exp(-thin*sqrt(sqr(x)+sqr(y))))
Draw(kernel,mesh,'K')

grid = GridFunction(fes)
Dr = GridFunction(fes)

# GridFunctions for caching of convolution values and automatic gradient calculation

with TaskManager():
    grid.Set(conv)
    
Dr.Set(1./10*exp(-cdec*grid))
   

a = BilinearForm(fes, symmetric=False)
a += SymbolicBFI((1-r2)*(grad(Dr)*r+Dr*grad(r))*grad(tr) + r2*Dr*grad(r)*grad(tr))

#Deps= 1
#a += SymbolicBFI(Dr*(Deps*((1-r2-b2)*grad(r) + r2*(grad(r)+grad(b))) + r2*(1-r2-b2)*(-cdec*grad(gridr)+cdec2*grad(gridb)))*grad(tr))
#a += SymbolicBFI(Db*(Deps*((1-r2-b2)*grad(b) + b2*(grad(r)+grad(b))) + b2*(1-r2-b2)*(-cdec*grad(gridb)+cdec2*grad(gridr)))*grad(tb))

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(r*tr)

print('Assembling m...')
m.Assemble()

rhs = r2.vec.CreateVector()
mstar = m.mat.CreateMatrix()

if netmesh.dim == 1:
    plt.figure('dynamic')
    rplot = Plot(r2, 'r')
    plt.show(block=False)
else:
    Draw(r2, mesh, 'r')
    # visualize both species at the same time, red in top mesh, blue in bottom
    # translate density b2 of blue species to bottom mesh
    #both = r2 + Compose((x, y-yoffset), b2, mesh)
    #bothgrid = gridr + Compose((x, y-yoffset), gridb, mesh)
    Draw(grid, mesh, 'G*r')
    Draw(Dr, mesh, 'Dr')
#    Draw(convr, mesh, 'convr')
#    Draw(convb, mesh, 'convb')

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[r2],names=["r"],filename="schellingdat_onespecies/schelling_",subdivision=3)
    vtk.Do()



# semi-implicit Euler
input("Press any key")
t = 0.0
k = 0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
#        print("\nt = {:10.6e}".format(t))
        t += tau
        k += 1

        if conv:
            grid.Set(conv)
            Dr.Set(1./10*(exp(-cdec*grid)))
        a.Assemble()

        rhs.data = m.mat * r2.vec

        mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse(fes.FreeDofs())
        r2.vec.data = invmat * rhs
        

        if netmesh.dim == 1 and k % 50 == 0:
            rplot.Redraw()
            plt.pause(0.05)
        elif k % 20 == 0:
            Redraw(blocking=False)
            if vtkoutput:
                vtk.Do()

        if k % 50 == 0:
            print("\n mass r = {:10.6e}".format(Integrate(r2,mesh)) + "t = {:10.6e}".format(t))

outfile.close()

