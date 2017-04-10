from ngsolve import *
from ngsapps.utils import *
import scipy.optimize as opt
import itertools as it

import numpy as np

class CrossDiffParams:
    def __init__(self, s=None, Dr=None, Db=None, Vr=None, Vb=None):
        self.s = s
        self.Dr = Dr
        self.Db = Db
        self.Vr = Vr
        self.Vb = Vb

def AApply(uv, p, mesh, mat):
    mmr = Integrate(exp((uv[0]-p.Vr)/p.Dr) / (1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db)), mesh, definedon=mat)
    mmb = Integrate(exp((uv[1]-p.Vb)/p.Db) / (1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db)), mesh, definedon=mat)
    # print(np.hstack((mmr, mmb)))
    return (np.hstack((mmr, mmb)))

def AssembleLinearization(uv, p, mesh, mat):
    m = np.empty([2,2])
    m[0,0] = Integrate(exp((uv[0]-p.Vr)/p.Dr)*(1+exp((uv[1]-p.Vb)/p.Db))/(sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db))*p.Dr),mesh,definedon=mat)
    m[0,1] = Integrate(-1/p.Db*exp((uv[0]-p.Vr)/p.Dr)*exp((uv[1]-p.Vb)/p.Db)/sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db)),mesh,definedon=mat)

    m[1,0] = Integrate(-1/p.Dr*exp((uv[0]-p.Vr)/p.Dr)*exp((uv[1]-p.Vb)/p.Db)/sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db)),mesh,definedon=mat)
    m[1,1] = Integrate(exp((uv[1]-p.Vb)/p.Db)*(1+exp((uv[0]-p.Vr)/p.Dr))/(sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db))*p.Db),mesh,definedon=mat)

    # print(m)
    return m

def stationarySols(p, fes, mat):
    mesh = fes.__getstate__()[1]

    # Calculate constant equilibria
    domainSize = Integrate(CoefficientFunction(1), mesh, definedon=mat)
    mr = Integrate(p.s.components[0], mesh, definedon=mat)
    mb = Integrate(p.s.components[1], mesh, definedon=mat)

    def J(uv):
        return AApply(uv, p, mesh, mat) - [mr, mb]
    def DJ(uv):
        return AssembleLinearization(uv, p, mesh, mat)

    rbinfty = GridFunction(fes)
    rinfty = rbinfty.components[0]
    binfty = rbinfty.components[1]

    print('Root finder...', end='')
    for x,y in it.product(np.arange(-1, 1, 0.1), repeat=2):
        optres = opt.root(J, [x,y], jac=DJ)
        if optres.success:
            print('converged')
            break
        print('.', end='')
    if not optres.success:
        print()
        raise RuntimeError('Root finder did not converge!')
    uvinfty = optres.x

    rinfty.Set(exp((uvinfty[0]-p.Vr)/p.Dr) / (1+exp((uvinfty[0]-p.Vr)/p.Dr)+exp((uvinfty[1]-p.Vb)/p.Db)))
    binfty.Set(exp((uvinfty[1]-p.Vb)/p.Db) / (1+exp((uvinfty[0]-p.Vr)/p.Dr)+exp((uvinfty[1]-p.Vb)/p.Db)))

    return rbinfty
