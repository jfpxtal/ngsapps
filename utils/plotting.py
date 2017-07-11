import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import math

class MPLMesh1D:
    def __init__(self, mesh, subdivision=1):
        h = 1/(subdivision+1)
        # need to keep trafos as member to prevent deallocation
        # because mips keep a reference to the trafo
        self.trafos = []
        els = []
        for e in mesh.Elements():
            trafo = mesh.GetTrafo(e)
            self.trafos.append(trafo)
            elmips = []
            for i in range(subdivision+2):
                mip = trafo(i*h)
                elmips.append(mip)
            elmips = sorted(elmips, key=lambda mip: mip.point[0])
            els.append(elmips)
        els = sorted(els, key=lambda el: el[0].point[0])
        self.px = []
        self.mips = []
        for e in els:
            for mip in e:
                self.px.append(mip.point[0])
                self.mips.append(mip)
            self.px.append(math.nan)
            self.mips.append(math.nan)

    def Plot(self, func, ax=None, *args, **kwargs):
        if not ax:
            ax = plt.gca()

        l = MPLLine(self)
        l.Draw(func, ax, *args, **kwargs)
        return l

class MPLMesh2D:
    def __init__(self, mesh, subdivision=1):
        r = 2**subdivision
        h = 1/r
        self.mips = []
        # need to keep trafos as member to prevent deallocation
        # because mips keep a reference to the trafo
        self.trafos = []
        px = []
        py = []
        triangles = []
        pidx = 0
        for e in mesh.Elements():
            trafo = mesh.GetTrafo(e)
            self.trafos.append(trafo)
            for i in range(r+1):
                for j in range(r-i+1):
                    mip = trafo(j*h, i*h)
                    px.append(mip.point[0])
                    py.append(mip.point[1])
                    self.mips.append(mip)
                    if i+j < r:
                        pidx_incr_i = pidx+1
                        pidx_incr_j = pidx+r+1-i
                        triangles.append([pidx, pidx_incr_i, pidx_incr_j])
                        pidx_incr_ij = pidx_incr_j+1
                        if i+j+1 < r:
                            triangles.append([pidx_incr_i, pidx_incr_ij, pidx_incr_j])

                    pidx += 1

        self.triang = Triangulation(px, py, triangles)

    def Plot(self, func, ax=None, *args, **kwargs):
        if not ax:
            ax = plt.gca()

        ts = MPLTriSurf(self)
        ts.Draw(func, ax, *args, **kwargs)
        return ts

class MPLLine:
    def __init__(self, mplmesh):
        self.mesh = mplmesh

    def GetValues(self, func):
        py = []
        for mip in self.mesh.mips:
            if mip is math.nan:
                py.append(math.nan)
            else:
                py.append(func(mip))
        return py

    def Draw(self, func, ax, *args, **kwargs):
        self.func = func
        self.ax = ax
        self.args = args
        self.kwargs = kwargs
        py = self.GetValues(func)
        self.line, = ax.plot(self.mesh.px, py, *args, **kwargs)

    def Redraw(self, autoscale=True):
        py = self.GetValues(self.func)
        self.line.set_ydata(py)
        if autoscale:
            self.ax.relim()
            self.ax.autoscale_view()

class MPLTriSurf:
    def __init__(self, mplmesh):
        self.mesh = mplmesh

    def Draw(self, func, ax, *args, **kwargs):
        self.func = func
        self.ax = ax
        self.args = args
        self.kwargs = kwargs
        pz = [func(mip) for mip in self.mesh.mips]
        self.surf = ax.plot_trisurf(self.mesh.triang, pz, *args, **kwargs)

    def Redraw(self, autoscale=True):
        self.surf.remove()
        self.Draw(self.func, self.ax, *self.args, **self.kwargs)
        if autoscale:
            self.ax.relim()
            self.ax.autoscale_view()

def Plot(func, *args, ax=None, mplmesh=None, mesh=None, subdivision=1, **kwargs):
    if not mplmesh:
        if not mesh:
            # only works for GridFunctions, not CoefficientFunctions
            mesh = func.space.mesh
        if mesh.dim == 1:
            mplmesh = MPLMesh1D(mesh, subdivision)
        else:
            mplmesh = MPLMesh2D(mesh, subdivision)

    if not ax:
        if type(mplmesh) is MPLMesh1D:
            ax = plt.gca()
        else:
            ax = plt.gca(projection='3d')

    return mplmesh.Plot(func, ax, *args, **kwargs)
