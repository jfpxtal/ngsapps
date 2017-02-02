from netgen.geom2d import SplineGeometry
from ngsolve import *

from ngsapps.utils import *
from ngsapps.merge_meshes import *
from ngsapps.meshtools import *

import geometries
from stationary import *
from dgform import DGFormulation
from cgform import CGFormulation

from itertools import product

orders = [1, 2, 3]
maxhs = [i*0.02 for i in range(3, 14)]
etas = [20, 25, 30, 50, 100, 200, 1000]
forms = [CGFormulation()] + [DGFormulation(eta) for eta in etas]
convs = [False, True]

p = CrossDiffParams()

# diffusion coefficients
# red species
p.Dr = 0.1
# blue species
p.Db = 0.3

# advection potentials
p.Vr = -x+sqr(y-0.5)
p.Vb = x+sqr(y-0.5)

conv = False

# time step and end
tau = 0.05
tend = 20

# geometry and mesh
geo = SplineGeometry()
doms = geometries.patchClamp(geo)
for d in range(1, doms+1):
    geo.SetMaterial(d, 'top')

for conv, order, maxh, form in product(convs, orders, maxhs, forms):
    curpars = 'order{}_maxh{}_form{}_conv{}'.format(order, maxh, form, conv)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(curpars)
    outfile = open('../data/crossdiff/topf/' + curpars + '.csv', 'w')
    outfile.write('time, entropy, l2sq_to_equi_r, l2sq_to_equi_b\n')

    # generate mesh on top geometry
    netmesh = geo.GenerateMesh(maxh=maxh)
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

    entropy = rinfty*ZLogZCF(r2/rinfty) + binfty*ZLogZCF(b2/binfty) + (1-rinfty-binfty)*ZLogZCF((1-r2-b2)/(1-rinfty-binfty)) + r2*gridr + b2*gridb
    ent = Integrate(entropy, mesh, definedon=topMat)
    l2r = Integrate(sqr(rinfty-r2), mesh, definedon=topMat)
    l2b = Integrate(sqr(binfty-b2), mesh, definedon=topMat)
    outfile.write('{}, {}, {}, {}\n'.format(0.0, ent, l2r, l2b))

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
            ent = Integrate(entropy, mesh, definedon=topMat)
            l2r = Integrate(sqr(rinfty-r2), mesh, definedon=topMat)
            l2b = Integrate(sqr(binfty-b2), mesh, definedon=topMat)
            outfile.write('{}, {}, {}, {}\n'.format(t, ent, l2r, l2b))
            outfile.flush()

    outfile.close()

