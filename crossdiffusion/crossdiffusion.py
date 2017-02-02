from netgen.geom2d import SplineGeometry
from ngsolve import *

from ngsapps.utils import *
from ngsapps.merge_meshes import *
from ngsapps.meshtools import *

import matplotlib.pyplot as plt

import geometries
from stationary import *
from dgform import DGFormulation
from cgform import CGFormulation


order = 3
maxh = 0.3

convOrder = 3

p = CrossDiffParams()

# diffusion coefficients
# red species
p.Dr = 0.1
# blue species
p.Db = 0.3

# advection potentials
p.Vr = -x+sqr(y-0.5)
p.Vb = x+sqr(y-0.5)

# time step and end
tau = 0.05
tend = -1

# jump penalty
eta = 50

form = CGFormulation()
# form = DGFormulation(eta)

conv = True

# geometry and mesh
geo = SplineGeometry()
doms = geometries.patchClamp(geo)
for d in range(1, doms+1):
    geo.SetMaterial(d, 'top')

# generate mesh on top geometry
netmesh = geo.GenerateMesh(maxh=maxh)

# now add a copy of the mesh, translated by yoffset, for visualization of species blue
yoffset = -1.3
netmesh = merge_meshes(netmesh, netmesh, offset2=(0, yoffset, 0), transfer_mats2=False)
for d in range(doms+1, nr_materials(netmesh)+1):
    netmesh.SetMaterial(d, 'bottom')

mesh = Mesh(netmesh)
topMat = mesh.Materials('top')

fes1, fes = form.FESpace(mesh, order)
r, b = fes.TrialFunction()
tr, tb = fes.TestFunction()

# initial values
p.s = GridFunction(fes)
r2 = p.s.components[0]
b2 = p.s.components[1]
# r2.Set(IfPos(0.2-x, IfPos(0.5-y, 0.9, 0), 0))
# b2.Set(IfPos(x-1.8, 0.6, 0))
r2.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)))
b2.Set(0.5*exp(-pow(x-1.9, 2)-0.1*pow(y-0.5, 2)))
#r2.Set(0.5+0*x)
#b2.Set(0.5+0*x)

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
line, = ax.plot(times, ents)
plt.show(block=False)

# input("Press any key...")
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

        Redraw(blocking=False)
        times.append(t)
        ents.append(Integrate(entropy, mesh, definedon=topMat))
        line.set_xdata(times)
        line.set_ydata(ents)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        # input()

outfile = open('ents_dg.csv','w')
for item in ents:
    outfile.write("%s\n" % item)
outfile.close()

