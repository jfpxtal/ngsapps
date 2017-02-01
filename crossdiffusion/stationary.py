from ngsolve import *
from ngsapps.utils import *

import numpy as np

class CrossDiffParams:
    pass
    # def __init__(self, rinit=None, binit=None, Dr=None, Db=None, Vr=None, Vb=None):
    #     self.rinit = rinit
    #     self.binit = binit
    #     self.Dr = Dr
    #     self.Db = Db
    #     self.Vr = Vr
    #     self.Vb = Vb

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

    rbinfty = GridFunction(fes)
    rinfty = rbinfty.components[0]
    binfty = rbinfty.components[1]

    #rinfty = Integrate(r2, mesh, definedon=mat) / domainSize
    #binfty = Integrate(b2, mesh, definedon=mat) / domainSize

    #### TODO: Add diffusion coeffs

    # Newton Solver to determine stationary solutions

    updnorm = 1e99
    uvinfty = np.hstack((0.0,0.0))
    # Newton solver
    print('Start Newton...')
    with TaskManager():
        while updnorm > 1e-9:
            rhs = AApply(uvinfty, p, mesh, mat) - np.hstack((mr, mb))
            Alin = AssembleLinearization(uvinfty, p, mesh, mat)
        #    hurgh
            upd = np.linalg.solve(Alin, rhs)

            updnorm = np.linalg.norm(upd)

            uvinfty = uvinfty - 0.1*upd
            # input('')

    print('Newton converged with error' + '|w| = {:7.3e} '.format(updnorm),end='\n')

    rinfty.Set(exp((uvinfty[0]-p.Vr)/p.Dr) / (1+exp((uvinfty[0]-p.Vr)/p.Dr)+exp((uvinfty[1]-p.Vb)/p.Db)))
    binfty.Set(exp((uvinfty[1]-p.Vb)/p.Db) / (1+exp((uvinfty[0]-p.Vr)/p.Dr)+exp((uvinfty[1]-p.Vb)/p.Db)))

    return rbinfty
