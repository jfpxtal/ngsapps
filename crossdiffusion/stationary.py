from ngsolve import *
from ngsapps.utils import *
import scipy.optimize as opt
import itertools as it

import numpy as np
import math

class CrossDiffParams:
    def __init__(self, s=None, Dr=None, Db=None, Vr=None, Vb=None):
        self.s = s
        self.Dr = Dr
        self.Db = Db
        self.Vr = Vr
        self.Vb = Vb

def max(x, y):
    return IfPos(x-y, x, y)

def AApply(uv, p, mesh, mat):
    a = (uv[0]-p.Vr)/p.Dr
    b = (uv[1]-p.Vb)/p.Db

    mmr = Integrate(1 / (1+exp(-a)+exp(b-a)), mesh, definedon=mat)
    mmb = Integrate(1 / (1+exp(-b)+exp(a-b)), mesh, definedon=mat)

    return (np.hstack((mmr, mmb)))

def AssembleLinearization(uv, p, mesh, mat):
    a = (uv[0]-p.Vr)/p.Dr
    b = (uv[1]-p.Vb)/p.Db

    m = np.empty([2,2])

    # For very small diffusion coeffs Dr or Db, the exponents get too large.
    # To prevent NaN's, we need to switch between equivalent expressions.

    m[0,0] = Integrate(IfPos(
        -max(a, b),
        exp(a)*(1+exp(b))/sqr(1+exp(a)+exp(b)),           # a <= 0 && b <= 0
        IfPos(
            a-b,
            (exp(-a)+exp(b-a))/sqr(1+exp(-a)+exp(b-a)),   # a >= b && a > 0
            exp(a-b)*(1+exp(-b))/sqr(1+exp(-b)+exp(a-b))  # b > a && b > 0
            )
        )/p.Dr, mesh, definedon=mat)

    m[1,1] = Integrate(IfPos(
        -max(a, b),
        exp(b)*(1+exp(a))/sqr(1+exp(a)+exp(b)),           # a <= 0 && b <= 0
        IfPos(
            a-b,
            exp(b-a)*(1+exp(-a))/sqr(1+exp(-a)+exp(b-a)), # a >= b && a > 0
            (exp(-b)+exp(a-b))/sqr(1+exp(-b)+exp(a-b))    # b > a && b > 0
            )
        )/p.Db, mesh, definedon=mat)

    m[0,1] = Integrate(-1/p.Db*IfPos(
        -max(a, b),
        exp(a+b)/sqr(1+exp(a)+exp(b)),        # a <= 0 && b <= 0
        IfPos(
            a-b,
            exp(b-a)/sqr(1+exp(-a)+exp(b-a)), # a >= b && a > 0
            exp(a-b)/sqr(1+exp(-b)+exp(a-b))  # b > a && b > 0
            )
        ), mesh, definedon=mat)

    m[1,0] = Integrate(-1/p.Dr*IfPos(
        -max(a, b),
        exp(a+b)/sqr(1+exp(a)+exp(b)),        # a <= 0 && b <= 0
        IfPos(
            a-b,
            exp(b-a)/sqr(1+exp(-a)+exp(b-a)), # a >= b && a > 0
            exp(a-b)/sqr(1+exp(-b)+exp(a-b))  # b > a && b > 0
            )
        ), mesh, definedon=mat)

    # m2 = np.empty([2,2])
    # m2[0,0] = Integrate(exp((uv[0]-p.Vr)/p.Dr)*(1+exp((uv[1]-p.Vb)/p.Db))/(sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db))*p.Dr),mesh,definedon=mat)
    # m2[0,1] = Integrate(-1/p.Db*exp((uv[0]-p.Vr)/p.Dr)*exp((uv[1]-p.Vb)/p.Db)/sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db)),mesh,definedon=mat)

    # m2[1,0] = Integrate(-1/p.Dr*exp((uv[0]-p.Vr)/p.Dr)*exp((uv[1]-p.Vb)/p.Db)/sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db)),mesh,definedon=mat)
    # m2[1,1] = Integrate(exp((uv[1]-p.Vb)/p.Db)*(1+exp((uv[0]-p.Vr)/p.Dr))/(sqr(1+exp((uv[0]-p.Vr)/p.Dr)+exp((uv[1]-p.Vb)/p.Db))*p.Db),mesh,definedon=mat)

    # err = np.linalg.norm(m-m2)
    # if err > 0:
    #     print(err)

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
            print('converged with initial guess {}'.format((x, y)))
            break
        print('.', end='')
    if not optres.success:
        print()
        raise RuntimeError('Root finder did not converge!')
    uvinfty = optres.x

    rinfty.Set(exp((uvinfty[0]-p.Vr)/p.Dr) / (1+exp((uvinfty[0]-p.Vr)/p.Dr)+exp((uvinfty[1]-p.Vb)/p.Db)))
    binfty.Set(exp((uvinfty[1]-p.Vb)/p.Db) / (1+exp((uvinfty[0]-p.Vr)/p.Dr)+exp((uvinfty[1]-p.Vb)/p.Db)))

    return rbinfty
