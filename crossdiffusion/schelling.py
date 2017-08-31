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
maxh = 0.02

conv_order = 3

# time step and end
tau = 0.005
tend = -1


p = CrossDiffParams()

# diffusion coefficients
# red species
#p.Dr = 0.05
# blue species
#p.Db = 0.15
# p.Dr = 0.0004
# p.Db = 0.0001

# advection potentials
#p.Vr = -x+sqr(y-0.5)
#p.Vb = x+sqr(y-0.5)


# jump penalty
#eta = 15

# form = CGFormulation()
#form = DGFormulation(eta)

conv = True

yoffset = -1.3
#netmesh = geos.make1DMesh(maxh)
#netmesh = geos.make2DMesh(maxh, yoffset, geos.square)
#
#mesh = Mesh(netmesh)

geo = SplineGeometry()
xmin, xmax = -1/2, 1/2
ymin, ymax = -1/2, 1/2
dx = xmax-xmin
dy = ymax-ymin
MakePeriodicRectangle(geo, (xmin, ymin), (xmax, ymax))
netmesh = geo.GenerateMesh(maxh=maxh)
mesh = Mesh(netmesh)

#topMat = mesh.Materials('top')

#fes1, fes = form.FESpace(mesh, order)
fes1 = Periodic(H1(mesh, order=order)) #, flags={'definedon': ['top']}))
# calculations only on top mesh
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
freq = 10
#r2.Set(2.0/3*0.5*(sin(freq*x)*sin(freq*y)+1))
#b2.Set(1.0/3*0.5*(cos(freq*x)*cos(freq*y)+1))

r2.Set(RandomCF(0, 0.49))
b2.Set(RandomCF(0, 0.49))
#r2.Set(0.5+0*x)
#b2.Set(0.5+0*x)
cdec = 10
cdec2 = 5
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
    K = CompactlySupportedKernel(0.05)

    #K = exp(-sqrt(sqr(x-xPar)*x+sqr(y-yPar)))
    convr = ParameterLF(fes1.TestFunction()*K, r2, conv_order, repeat=0, patchSize=[dx, dy])
    convb = ParameterLF(fes1.TestFunction()*K, b2, conv_order, repeat=0, patchSize=[dx, dy])
else:
    asd
    convr = 0
    convb = 0
    

kernel = GridFunction(fes1)
kernel.Set(k0*exp(-thin*sqrt(sqr(x)+sqr(y))))
Draw(kernel,mesh,'K')

grid = GridFunction(fes)
gridr = grid.components[0]
gridb = grid.components[1]

tmp = GridFunction(fes)
Dr = tmp.components[0]
Db = tmp.components[1]

# GridFunctions for caching of convolution values and automatic gradient calculation

with TaskManager():
    gridr.Set(convr)
    gridb.Set(convb)
    
Dr.Set(1./10*(exp(-cdec*gridr+cdec2*gridb)))
Db.Set(1./10*(exp(cdec2*gridr-cdec*gridb)))
   

a = BilinearForm(fes, symmetric=False)
a += SymbolicBFI((1-r2-b2)*(grad(Dr)*r+Dr*grad(r))*grad(tr) + r2*Dr*(grad(r)+grad(b))*grad(tr))
a += SymbolicBFI((1-r2-b2)*(grad(Db)*b+Db*grad(b))*grad(tb) + b2*Db*(grad(r)+grad(b))*grad(tb))
#Deps= 1
#a += SymbolicBFI(Dr*(Deps*((1-r2-b2)*grad(r) + r2*(grad(r)+grad(b))) + r2*(1-r2-b2)*(-cdec*grad(gridr)+cdec2*grad(gridb)))*grad(tr))
#a += SymbolicBFI(Db*(Deps*((1-r2-b2)*grad(b) + b2*(grad(r)+grad(b))) + b2*(1-r2-b2)*(-cdec*grad(gridb)+cdec2*grad(gridr)))*grad(tb))

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(r*tr)
m += SymbolicBFI(b*tb)

print('Assembling m...')
m.Assemble()

rhs = p.s.vec.CreateVector()
mstar = m.mat.CreateMatrix()

if netmesh.dim == 1:
    plt.figure('dynamic')
    rplot = Plot(r2, 'r')
    bplot = Plot(b2, 'b')
    plt.show(block=False)
else:
    Draw(r2, mesh, 'r')
    Draw(b2, mesh, 'b')
    # visualize both species at the same time, red in top mesh, blue in bottom
    # translate density b2 of blue species to bottom mesh
    #both = r2 + Compose((x, y-yoffset), b2, mesh)
    #bothgrid = gridr + Compose((x, y-yoffset), gridb, mesh)
    Draw(gridr, mesh, 'G*r')
    Draw(gridb, mesh, 'G*b')
    Draw(Dr, mesh, 'Dr')
    Draw(Db, mesh, 'Db')
#    Draw(convr, mesh, 'convr')
#    Draw(convb, mesh, 'convb')

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
            gridr.Set(convr)
            gridb.Set(convb)  # print('Assembling a...')
            Dr.Set(1./10*(exp(-cdec*gridr+cdec2*gridb)))
            Db.Set(1./10*(exp(cdec2*gridr-cdec*gridb))) 
        a.Assemble()

        rhs.data = m.mat * p.s.vec

        mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse(fes.FreeDofs())
        p.s.vec.data = invmat * rhs
        
        r2 = p.s.components[0]
        b2 = p.s.components[1]

        if netmesh.dim == 1:
            rplot.Redraw()
            bplot.Redraw()
            plt.pause(0.05)
        elif k % 1 == 0:
            Redraw(blocking=False)

        input("")
            
        print("\n mass r = {:10.6e}".format(Integrate(r2,mesh)) +  " mass b = {:10.6e}".format(Integrate(b2,mesh)) + "t = {:10.6e}".format(t))
        

#        ent = Integrate(entropy, mesh, definedon=topMat)
#        l2r = Integrate(sqr(rinfty-r2), mesh, definedon=topMat)
#        l2b = Integrate(sqr(binfty-b2), mesh, definedon=topMat)
#        outfile.write('{}, {}, {}, {}\n'.format(t, ent, l2r, l2b))
#        outfile.flush()

#        times.append(t)
#        ents.append(ent)
#        line.set_xdata(times)
#        line.set_ydata(ents)
#        ax.relim()
#        ax.autoscale_view()
#        fig.canvas.draw()

outfile.close()

