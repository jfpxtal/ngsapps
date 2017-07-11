from ngsolve import *
from ngsolve.comp import Region
from ngsapps.plotting import *
import matplotlib.pyplot as plt

def sign(x):
    if x < 0:
        return -1
    else:
        return int(x > 0)

def minmod(a1, a2, a3, h):
    mu = 1

    if abs(a1) <= mu*h**2:
        return a1
    elif sign(a1) == sign(a2) == sign(a3):
        return sign(a1)*min(abs(a1), abs(a2), abs(a3))
    else:
        return 0

def limitValues(leftElVals, thisElVals, rightElVals, size):
    newLVal = thisElVals['avg'] - minmod(thisElVals['avg']-thisElVals['lval'],
                                         thisElVals['avg']-leftElVals['avg'],
                                         rightElVals['avg']-thisElVals['avg'], size)
    newRVal = thisElVals['avg'] + minmod(thisElVals['rval']-thisElVals['avg'],
                                         thisElVals['avg']-leftElVals['avg'],
                                         rightElVals['avg']-thisElVals['avg'], size)
    return newLVal, newRVal

def stabilityLimiter(g, dgform, mat, plot):
    fes = g.__reduce__()[1][0]
    mesh = fes.mesh
    p1fes,_ = dgform.FESpace(mesh, mat, 1)
    p1gf = GridFunction(p1fes)
    p1gf.Set(g)
    ints = Integrate(g, mesh, element_wise=True)
    els = []
    for e in fes.Elements():
        trafo = e.GetTrafo()
        elmips = sorted([trafo(0), trafo(1)], key=lambda mip: mip.point[0])
        size = elmips[1].point[0]-elmips[0].point[0]
        lval, rval = g(elmips[0]), g(elmips[1])
        p1lval, p1rval = p1gf(elmips[0]), p1gf(elmips[1])
        els.append({'dofs': e.dofs,
                    'midpoint': trafo(0.5).point[0],
                    'left': elmips[0].point[0],
                    'orig': {
                        'lval': lval,
                        'rval': rval,
                        'avg': ints[e.nr]/size
                        },
                    'p1': {
                        'lval': p1lval,
                        'rval': p1rval,
                        'avg': (p1lval+p1rval)/2
                        },
                    'size': size})

    els = sorted(els, key=lambda el: el['midpoint'])

    setgf = GridFunction(fes)

    # TODO: think about boundary conditions
    for i in range(1, len(els)-1):
        # input()
        el = els[i]

        # higher order
        testlval, testrval = limitValues(els[i-1]['orig'], els[i]['orig'], els[i+1]['orig'], els[i]['size'])
        if testlval != els[i]['orig']['lval'] or testrval != els[i]['orig']['rval']:
            newlval, newrval = limitValues(els[i-1]['p1'], els[i]['p1'], els[i+1]['p1'], els[i]['size'])
            setgf.Set(newlval + (x-els[i]['left'])*(newrval-newlval)/els[i]['size'],
                      definedon=Region(mesh, VOL, 'top'+str(i+1)))

            for d in el['dofs']:
                g.vec[d] = setgf.vec[d]

        # plot.Redraw()
        # plt.pause(0.05)

from ngsolve.fem import BaseMappedIntegrationPoint

def nonnegativityLimiter(g, dgform, mat, plot):
    fes = g.__reduce__()[1][0]
    mesh = fes.mesh
    p1fes,_ = dgform.FESpace(mesh, mat, 1)
    p1gf = GridFunction(p1fes)
    p1gf.Set(g)
    setgf = GridFunction(fes)
    els = []
    for i, e in enumerate(fes.Elements()):
        trafo = e.GetTrafo()
        elmips = sorted([trafo(0), trafo(1)], key=lambda mip: mip.point[0])
        lval, rval = g(elmips[0]), g(elmips[1])

        ir = IntegrationRule(e.type, fes.globalorder)
        negative = False
        for p in ir:
            if g(trafo(p)) < 0:
                negative = True
                break
        if negative or lval < 0 or rval < 0:
            # print('nonnegativityLimiter', e.nr)
            # input()
            # setgf.Set(p1gf,
            #         definedon=Region(mesh, VOL, 'top'+str(i+1)))
            # for d in e.dofs:
            #     g.vec[d] = setgf.vec[d]

            # plot.Redraw()
            # plt.pause(0.05)
            # input()
            midpoint = trafo(0.5).point[0]
            size = elmips[1].point[0]-elmips[0].point[0]
            p1lval, p1rval = p1gf(elmips[0]), p1gf(elmips[1])
            avg = (p1lval + p1rval) / 2
            if p1lval < 0:
                setgf.Set((1+2/size*(x-midpoint))*avg,
                        definedon=Region(mesh, VOL, 'top'+str(i+1)))
            elif p1rval < 0:
                setgf.Set((1-2/size*(x-midpoint))*avg,
                        definedon=Region(mesh, VOL, 'top'+str(i+1)))

            for d in e.dofs:
                g.vec[d] = setgf.vec[d]

            # plot.Redraw()
            # plt.pause(0.05)
