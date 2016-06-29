import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, \
                                       FaceDescriptor, Mesh as NetMesh
from netgen.csg import Pnt
from ngsolve import *

f = open('precip2d.bin', 'rb')
L = np.load(f)
dt = np.load(f)

ss = []
masses = []
while True:
    try:
        s = np.load(f)
        masses.append(s.sum())
        ss.append(s)
    except OSError:
        break

f.close()

tend = dt * (len(ss) - 1)
N = int(math.sqrt(len(ss[0]) / 2)-1)
print('N = %d, L = %.2f, dt = %.2f, tend = %.2f' % (N, L, dt, tend))
continuous_ngplot = False


netmesh = NetMesh()
netmesh.dim = 2

if continuous_ngplot:
    M = N
else:
    M = N+1

pnums = []
for i in range(M + 1):
    for j in range(M + 1):
        pnums.append(netmesh.Add(MeshPoint(Pnt(L * i / M, L * j / M, 0))))

netmesh.Add (FaceDescriptor(surfnr=1,domin=1,bc=1))
netmesh.SetMaterial(1, 'mat')
for j in range(M):
    for i in range(M):
        netmesh.Add(Element2D(1, [pnums[i + j * (M + 1)],
                                  pnums[i + (j + 1) * (M + 1)],
                                  pnums[i + 1 + (j + 1) * (M + 1)],
                                  pnums[i + 1 + j * (M + 1)]]))

    netmesh.Add(Element1D([pnums[M + j * (M + 1)],
                           pnums[M + (j + 1) * (M + 1)]], index=1))
    netmesh.Add(Element1D([pnums[0 + j * (M + 1)],
                           pnums[0 + (j + 1) * (M + 1)]], index=1))

for i in range(M):
    netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))
    netmesh.Add(Element1D([pnums[i + M * (M + 1)],
                           pnums[i + 1 + M * (M + 1)]], index=1))

mesh = Mesh(netmesh)
if continuous_ngplot:
    Vvis = H1(mesh, order=1)
else:
    Vvis = L2(mesh, order=0)
fes = FESpace([Vvis, Vvis])
svis = GridFunction(fes)
svis.vec.FV().NumPy()[:] = ss[0]
Draw(svis.components[0], mesh, 'c')
Draw(svis.components[1], mesh, 'e')

ts = np.linspace(0, tend, num=len(masses))
fig_mass, (ax_mass, ax_slider) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[10, 1]})
line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
ax_mass.legend()

slider = Slider(ax_slider, "Time", 0, tend, valinit=0)

tline = ax_mass.axvline(0, color='r')

def update(t):
    tline.set_xdata(t)
    t = int(t / dt)
    svis.vec.FV().NumPy()[:] = ss[t]
    fig_mass.canvas.draw_idle()
    Redraw(blocking=False)


slider.on_changed(update)

plt.show()
