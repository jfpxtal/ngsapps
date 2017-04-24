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

def CreateBilinearForm(fes, s, gridr,gridb):
    r, b = fes.TrialFunction()
    tr, tb = fes.TestFunction()
    r2 = s.components[0]
    b2 = s.components[1]
    Dr = 0.1
    Db = 0.1

    a = BilinearForm(fes, symmetric=False)
    a += SymbolicBFI(Dr*((1-r2-b2)*(grad(gridr)*r+gridr*grad(r))*grad(tr) + r2*gridr*(grad(r)+grad(b))*grad(tr)))
    a += SymbolicBFI(Db*((1-r2-b2)*(grad(gridb)*b+gridb*grad(b))*grad(tb) + b2*gridb*(grad(r)+grad(b))*grad(tb)))
#    a += SymbolicBFI(0.1*(grad(r)*grad(tr)+grad(b)*grad(tb))) # Regularization
    return a

order = 3
maxh = 0.1

convOrder = 3

# time step and end
tau = 0.001
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
netmesh = geos.make2DMesh(maxh, yoffset, geos.square)

mesh = Mesh(netmesh)
topMat = mesh.Materials('top')

#fes1, fes = form.FESpace(mesh, order)
fes1 = H1(mesh, order=order, flags={'definedon': ['top']})
# calculations only on top mesh
fes = FESpace([fes1, fes1], flags={'definedon': ['top']})

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
r2.Set(0.2*(sin(freq*x)*sin(freq*y)+1))
b2.Set(0.2*(cos(freq*x)*cos(freq*y)+1))
#r2.Set(0.5+0*x)
#b2.Set(0.5+0*x)
cdec = 20
cdec2 = 0.1

if conv:
    # convolution
#    thin = 200
#    k0 = 20
#    K = k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
    K = exp(-sqrt(x*x+y*y))
    convr = ParameterLF(fes1.TestFunction()*K, r2, convOrder)
    convb = ParameterLF(fes1.TestFunction()*K, b2, convOrder)
else:
    convr = 0
    convb = 0

# GridFunctions for caching of convolution values and automatic gradient calculation
grid = GridFunction(fes)
gridr = grid.components[0]
gridb = grid.components[1]

with TaskManager():
    gridr.Set(exp(-convr)+exp(convb))
    gridb.Set(exp(convr)+exp(-convb))

a = BilinearForm(fes, symmetric=False)
a += SymbolicBFI((1-r2-b2)*(grad(gridr)*r+gridr*grad(r))*grad(tr) + r2*gridr*(grad(r)+grad(b))*grad(tr))
a += SymbolicBFI((1-r2-b2)*(grad(gridb)*b+gridb*grad(b))*grad(tb) + b2*gridb*(grad(r)+grad(b))*grad(tb))

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
    # Draw(r2, mesh, 'r')
    # Draw(b2, mesh, 'b')
    # visualize both species at the same time, red in top mesh, blue in bottom
    # translate density b2 of blue species to bottom mesh
    both = r2 + Compose((x, y-yoffset), b2, mesh)
    Draw(both, mesh, 'dynamic')


#times = [0.0]
#entropy = rinfty*ZLogZCF(r2/rinfty) + binfty*ZLogZCF(b2/binfty) + (1-rinfty-binfty)*ZLogZCF((1-r2-b2)/(1-rinfty-binfty)) + r2*gridr + b2*gridb
#ents = [Integrate(entropy, mesh, definedon=topMat)]
#fig, ax = plt.subplots()
#fig.canvas.set_window_title('entropy')
#line, = ax.plot(times, ents)
#plt.show(block=False)

#outfile = open('order{}_maxh{}_form{}_conv{}.csv'.format(order, maxh, form, conv), 'w')
#outfile.write('time, entropy, l2sq_to_equi_r, l2sq_to_equi_b\n')


# semi-implicit Euler
input("Press any key")
t = 0.0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
#        print("\nt = {:10.6e}".format(t))
        t += tau

        if conv:
#            print('Calculating convolution integrals...')
            gridr.Set(exp(-cdec*convr)+exp(cdec2*convb))
            gridb.Set(exp(cdec2*convr)+exp(-cdec*convb))
#        print('Assembling a...')
        a.Assemble()

        rhs.data = m.mat * p.s.vec

        mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse(fes.FreeDofs())
        p.s.vec.data = invmat * rhs

        if netmesh.dim == 1:
            rplot.Redraw()
            bplot.Redraw()
            plt.pause(0.05)
        else:
            Redraw(blocking=False)
            
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

