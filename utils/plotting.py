from matplotlib.tri import Triangulation

class MPLTriSurf:
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
        self.surf = None

    def Draw(self, func, ax, *args, **kwargs):
        pz = [func(mip) for mip in self.mips]
        if self.surf:
            self.surf.remove()
        self.surf = ax.plot_trisurf(self.triang, pz, *args, **kwargs)
