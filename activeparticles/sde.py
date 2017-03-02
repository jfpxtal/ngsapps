from math import sin, cos, tan, sqrt, pi
import numpy as np
import numpy.random as rnd
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import scipy.spatial.distance as sd
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
# import os
# # os.system("taskset -p 0xff %d" % os.getpid())
# print(os.sched_getaffinity(0))
# os.sched_setaffinity(0, {0, 1, 2, 3})
# print(os.sched_getaffinity(0))

N = 1000

# diameter of particles
sigma = 5

xmin = 0
xmax = 2000*sigma
ymin = 0
ymax = 485*sigma
Lx = xmax-xmin
Ly = ymax-ymin

# Lennard-Jones energy
epsilon = 1 # ?

# Lennard-Jones time
tauLJ = 1 # ?

# maximum swim speed
v0 = 24*sigma/tauLJ

# active Péclet number
# Pe = 500
Pe = 1e-3

# rotational diffusion coefficient
DR = 3*v0/(sigma*Pe)

# translational diffusion coefficient
DT = sigma**2*DR/3

# 1/(kB*T)
beta = epsilon*Pe/24

# time step
dt = 1e-4 * tauLJ

def vec(x, y):
    return np.array([x, y])

# force from shifted Lennard-Jones potential
# should this be periodic, i.e. consider the shortest distance across boundaries?
def F(points):
    rsq = sd.pdist(points, 'sqeuclidean')
    rsq = ma.masked_greater(rsq, 2**(1/3)*sigma**2)
    q = 4*epsilon*(-12*sigma**12/rsq**7 + 6*sigma**6/rsq**4)
    xs = points[:,0]
    ys = points[:,1]
    sf = sd.squareform(q.filled(0))
    xout = np.subtract.outer(xs, xs)
    yout = np.subtract.outer(ys, ys)
    return np.array([np.sum(xout*sf, axis=1), np.sum(yout*sf, axis=1)]).T

# def F(i, points):
#     Fx = 0
#     Fy = 0
#     for j in range(N):
#         if j == i:
#             continue
#         rsq = (points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2
#         if rsq > 2**(1/3)*sigma**2:
#             continue
#         q = 4*epsilon*(-12*sigma**12/rsq**7 + 6*sigma**6/rsq**4)
#         Fx += (points[i][0] - points[j][0]) * q
#         Fy += (points[i][1] - points[j][1]) * q
#     return vec(Fx, Fy)

def makeV(apex, angle, openingAngle, length, thickness):
    verts = [apex]
    verts.append(verts[-1] + length*vec(cos(angle-openingAngle/2), sin(angle-openingAngle/2)))
    verts.append(verts[-1] + thickness*vec(cos(angle-openingAngle/2+pi/2), sin(angle-openingAngle/2+pi/2)))
    verts.append(verts[-1] - (length-thickness/tan(openingAngle/2))*vec(cos(angle-openingAngle/2), sin(angle-openingAngle/2)))
    verts.append(verts[-1] + (length-thickness/tan(openingAngle/2))*vec(cos(angle+openingAngle/2), sin(angle+openingAngle/2)))
    verts.append(verts[-1] + thickness*vec(cos(angle+openingAngle/2+pi/2), sin(angle+openingAngle/2+pi/2)))
    return verts

polys = []
for i in range(3):
    for j in range(8):
        polys.append(makeV((200*sigma + i*280*sigma, j*(60+9)*sigma), 0, pi/3, 60*sigma, 10*sigma))
        polys.append(makeV((xmax - 200*sigma - i*280*sigma, j*(60+9)*sigma), pi, pi/3, 60*sigma, 10*sigma))
path = Path.make_compound_path_from_polys(np.array(polys))

# local swim speed
def v(i, inPoly):
    if inPoly[i]:
        return 0
    else:
        return v0

thetas = np.zeros(N)
xs = xmin + Lx*rnd.rand(N)
ys = ymin + Ly*rnd.rand(N)
points = np.array(list(zip(xs, ys)))

fig, ax = plt.subplots()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# TODO: markersize is in pixels
line, = plt.plot(xs, ys, 'ro', markersize=sigma)
ax.add_patch(PathPatch(path, alpha=0.3))

plt.show(block=False)

def doPoint(i, thetas, points, inPoly, forces):
    # TODO: precompute rnd.normal() for all points
    thetas[i] +=  sqrt(2*DR*dt) * rnd.normal()
    points[i] += v(i, inPoly)*vec(cos(thetas[i]), sin(thetas[i]))*dt + beta*DT*forces[i]*dt + sqrt(2*DT*dt)*rnd.normal(size=2)
    points[i][0] = xmin + (points[i][0]-xmin)%Lx
    points[i][1] = ymin + (points[i][1]-ymin)%Ly

k = 0
with Parallel(n_jobs=4) as parallel:
    while True:
        inPoly = path.contains_points(points)
        forces = F(points)
        parallel(delayed(has_shareable_memory)(doPoint(i, thetas, points, inPoly, forces)) for i in range(N))
        # [doPoint(i, thetas, points, inPoly, forces) for i in range(N)]

        if k % 100 == 0:
            line.set_data(points[:,0], points[:,1])
            fig.canvas.draw()
            input('ke')

        k += 1
