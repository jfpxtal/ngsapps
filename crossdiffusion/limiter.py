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

def stabilityLimiter(g, p):
    fes = g.__getstate__()[0]
    mesh = fes.__getstate__()[1]
    ints = Integrate(g, mesh, element_wise=True)
    els = []
    for e in fes.Elements():
        trafo = e.GetTrafo()
        elmips = sorted([trafo(0), trafo(1)], key=lambda mip: mip.point[0])
        size = elmips[1].point[0]-elmips[0].point[0]
        left, right = g(elmips[0]), g(elmips[1])
        els.append({'dofs': e.dofs,
                    'midpoint': trafo(0.5).point[0],
                    'left': left,
                    'right': right,
                    'avg': ints[e.nr]/size,
                    'slope': (right-left)/size,
                    'size': size})

    els = sorted(els, key=lambda el: el['midpoint'])

    setgf = GridFunction(fes)
    # TODO: think about boundary conditions
    for i in range(1, len(els)-1):
        # input()
        el = els[i]
        setgf.Set(el['avg'] + (x-el['midpoint'])*minmod(el['slope'],
                                                    (el['avg']-els[i-1]['avg'])/(el['size']/2),
                                                    (els[i+1]['avg']-el['avg'])/(el['size']/2), el['size']),
              definedon=Region(mesh, VOL, 'top'+str(i+1)))

        for d in el['dofs']:
            g.vec[d] = setgf.vec[d]
        # p.Redraw()
        # plt.pause(0.05)

        # higher order
        ## right = el.avg + minmod(el.right-el.avg, el.avg-els[i-1].avg, els[i+1].avg-el.avg)
        ## left = el.avg - minmod(el.right-el.avg, el.avg-els[i-1].avg, els[i+1].avg-el.avg)
        ## if left != el.left or right != el.right:
        ##     # l2 project etc


def nonnegativityLimiter():
    pass
