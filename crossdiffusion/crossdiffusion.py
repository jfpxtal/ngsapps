from netgen.geom2d import SplineGeometry
from ngsolve import *

from ngsapps.utils import *
from ngsapps.plotting import *

import matplotlib.pyplot as plt

import geometries as geos
from stationary import *
from limiter import *
from dgform import DGFormulation
from cgform import CGFormulation


order = 1
maxh = 0.1

convOrder = 3

p = CrossDiffParams()

# diffusion coefficients
# red species
p.Dr = 0.01
# blue species
p.Db = 0.03
# p.Dr = 0.004
# p.Db = 0.001

# advection potentials
# p.Vr = -x+sqr(y-0.5)
# p.Vb = x+sqr(y-0.5)
p.Vr = p.Vb = IfPos(x-0.5, sqr(x-0.5), IfPos(x+0.5, 0, sqr(x+0.5)))

# time step and end
tau = 0.005
tend = -1

# jump penalty
eta = 10

# form = CGFormulation()
form = DGFormulation(eta)

conv = False

# geometry and mesh
geo = SplineGeometry()
doms = geometries.window(geo)
for d in range(1, doms+1):
    geo.SetMaterial(d, 'top')

# generate mesh on top geometry
netmesh = geo.GenerateMesh(maxh=maxh)

# now add a copy of the mesh, translated by yoffset, for visualization of species blue
yoffset = -1.3
netmesh = geos.make1DMesh(maxh)
# netmesh = geos.make2DMesh(maxh, yoffset, geos.square)

mesh = Mesh(netmesh)
topMat = mesh.Materials('top')

Plot(p.Vr, mesh=mesh)
plt.figure()

fes1, fes = form.FESpace(mesh, order)
r, b = fes.TrialFunction()
tr, tb = fes.TestFunction()

# initial values
p.s = GridFunction(fes)
r2 = p.s.components[0]
b2 = p.s.components[1]
# r2.Set(IfPos(0.2-x, IfPos(0.5-y, 0.9, 0), 0))
# b2.Set(IfPos(x-1.8, 0.6, 0))
# r2.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)))
# b2.Set(0.5*exp(-pow(x-1.9, 2)-0.1*pow(y-0.5, 2)))
#r2.Set(0.5+0*x)
#b2.Set(0.5+0*x)
r2.Set(RandomCF(0, 0.49))
b2.Set(RandomCF(0, 0.49))

if conv:
    # convolution
    thin = 200
    k0 = 20
    K = k0*exp(-thin*(sqr(x-xPar)+sqr(y-yPar)))
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
    gridr.Set(p.Vr-convr)
    gridb.Set(p.Vb-convb)
velocities = (-grad(gridr),
              -grad(gridb))

a = form.BilinearForm(p, velocities)

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(r*tr)
m += SymbolicBFI(b*tb)

print('Assembling m...')
m.Assemble()

rbinfty = stationarySols(p, fes, topMat)
rinfty = rbinfty.components[0]
binfty = rbinfty.components[1]


rhs = p.s.vec.CreateVector()
mstar = m.mat.CreateMatrix()

if netmesh.dim == 1:
    plt.gcf().canvas.set_window_title('stationary')
    Plot(rinfty, 'r', subdivision=0)
    Plot(binfty, 'b', subdivision=0)
    plt.figure('dynamic')
    rplot = Plot(r2, 'r', subdivision=0)
    bplot = Plot(b2, 'b', subdivision=0)
    plt.show(block=False)
else:
    # Draw(r2, mesh, 'r')
    # Draw(b2, mesh, 'b')
    # visualize both species at the same time, red in top mesh, blue in bottom
    # translate density b2 of blue species to bottom mesh
    both = r2 + Compose((x, y-yoffset), b2, mesh)
    both2 = rinfty + Compose((x, y-yoffset), binfty, mesh)
    Draw(both2, mesh, 'stationary')
    Draw(both, mesh, 'dynamic')


times = [0.0]
entropy = rinfty*ZLogZCF(r2/rinfty) + binfty*ZLogZCF(b2/binfty) + (1-rinfty-binfty)*ZLogZCF((1-r2-b2)/(1-rinfty-binfty)) + r2*gridr + b2*gridb
ents = [Integrate(entropy, mesh, definedon=topMat)]
fig, ax = plt.subplots()
fig.canvas.set_window_title('entropy')
line, = ax.plot(times, ents)
plt.show(block=False)

outfile = open('order{}_maxh{}_form{}_conv{}.csv'.format(order, maxh, form, conv), 'w')
outfile.write('time, entropy, l2sq_to_equi_r, l2sq_to_equi_b\n')

input("Press any key...")
# semi-implicit Euler
t = 0.0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
        print("\nt = {:10.6e}".format(t))
        t += tau

        if conv:
            print('Calculating convolution integrals...')
            gridr.Set(p.Vr-convr)
            gridb.Set(p.Vb-convb)
        print('Assembling a...')
        a.Assemble()

        rhs.data = m.mat * p.s.vec

        mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse(fes.FreeDofs())
        p.s.vec.data = invmat * rhs

        # flux limiters
        # stabilityLimiter(r2, mesh, rplot)
        # stabilityLimiter(b2, mesh, bplot)

        if netmesh.dim == 1:
            rplot.Redraw()
            bplot.Redraw()
            plt.pause(0.05)
        else:
            Redraw(blocking=False)

        ent = Integrate(entropy, mesh, definedon=topMat)
        l2r = Integrate(sqr(rinfty-r2), mesh, definedon=topMat)
        l2b = Integrate(sqr(binfty-b2), mesh, definedon=topMat)
        outfile.write('{}, {}, {}, {}\n'.format(t, ent, l2r, l2b))
        outfile.flush()

        times.append(t)
        ents.append(ent)
        line.set_xdata(times)
        line.set_ydata(ents)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

outfile.close()

