# http://www.math.umn.edu/~scheel/preprints/pf0.pdf

import numpy as np
import numpy.random as random
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import threading
from queue import Queue

# from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
# from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor, Mesh as NetMesh
# from netgen.csg import Pnt

# from ngsapps.utils import *
np.set_printoptions(linewidth=400, threshold=100000)

# L = 10
# N = 4
L = 700
N = 70
dx = L / N

dt = 0.05
tend = 4000
# tend = 1000

gamma = 0.1
alpha = 0.2
kappa = 0

delta = 0.1

output = True
if output:
    outfile = open("precip.bin", "wb")

rowidxs = []
colidxs =[]
data = []

def to_vec_index(func, coord):
    # c00, c10, c20, ... , cN0 , c01, ... , cNN, e00, ...
    i, j = coord
    if func == 'c':
        return i + j * (N + 1)
    else:
        return (N + 1) ** 2 + i + j * (N + 1)

def newentry(i, j, d):
    # B[i, j] = d
    rowidxs.append(i)
    colidxs.append(j)
    data.append(d)

def add_laplace(func, scale, a, b, c):
    newentry(to_vec_index(func, b), to_vec_index(func, a), scale / dx ** 2)
    newentry(to_vec_index(func, b), to_vec_index(func, b), -2 * scale / dx ** 2)
    newentry(to_vec_index(func, b), to_vec_index(func, c), scale / dx ** 2)

for i in range(N + 1):
    for j in range(N + 1):
        if i > 0 and i < N:
            add_laplace('c', 1, (i-1,j), (i,j), (i+1,j))
            add_laplace('e', kappa, (i-1,j), (i,j), (i+1,j))
        if j > 0 and j < N:
            add_laplace('c', 1, (i,j-1), (i,j), (i,j+1))
            add_laplace('e', kappa, (i,j-1), (i,j), (i,j+1))

        newentry(to_vec_index('c', (i,j)), to_vec_index('c', (i,j)), -gamma)
        newentry(to_vec_index('e', (i,j)), to_vec_index('c', (i,j)), gamma)

def add_neumann(func, scale, bnd, inner):
    newentry(to_vec_index(func, bnd), to_vec_index(func, bnd), -2 * scale / dx ** 2)
    newentry(to_vec_index(func, bnd), to_vec_index(func, inner), 2 * scale / dx ** 2)

for i in range(N + 1):
    for f, s in [('c', 1), ('e', kappa)]:
        add_neumann(f, s, (i,0), (i,1))
        add_neumann(f, s, (i,N), (i,N-1))
        add_neumann(f, s, (0,i), (1,i))
        add_neumann(f, s, (N,i), (N-1,i))

B = sp.coo_matrix((data, (rowidxs, colidxs)), shape=(2 * (N + 1) ** 2, 2 * (N + 1) ** 2))
B *= -dt
B += sp.eye(2 * (N + 1) ** 2)
B = B.tocsr()
# print(B.todense())


def AApply(u):
    v = u[(N + 1) ** 2:]
    w = v * (1 - v) * (v - alpha)
    # print(dt * np.hstack((w, -w)))
    return dt * np.hstack((w, -w))

def AssembleLinearization(u):
    rightm = sp.dia_matrix((-3 * u[(N + 1) ** 2:] ** 2 + 2 * (1 + alpha) * u[(N + 1) ** 2:] - alpha, 0), ((N + 1) ** 2, (N + 1) ** 2))
    Alin = sp.bmat([[sp.coo_matrix(((N + 1) ** 2, (N + 1) ** 2)), rightm], [None, -rightm]])
    # print(dt * Alin.toarray())
    return dt * Alin


# s = random.rand(2 * (N + 1) ** 2)
# s = np.hstack((np.full(2 * (N + 1), delta), np.full(2 * (N + 1), -delta), np.zeros((N + 1) ** 2 - 4 * (N + 1)), np.full((N + 1) ** 2, alpha)))
s = np.hstack((np.full(10 * (N + 1), delta), np.full(10 * (N + 1), -delta), np.zeros((N + 1) ** 2 - 20 * (N + 1)), np.full((N + 1) ** 2, alpha)))

# netmesh = NetMesh()
# netmesh.dim = 2
# pnums = []
# for i in range(N + 1):
#     for j in range(N + 1):
#         pnums.append(netmesh.Add(MeshPoint(Pnt(L * i / N, L * j / N, 0))))

# print("b")
# netmesh.Add(FaceDescriptor())
# netmesh.SetMaterial(1, "mat")
# for j in range(N):
#     for i in range(N):
#         netmesh.Add(Element2D(1, [pnums[i + j * (N + 1)], pnums[i + (j + 1) * (N + 1)], pnums[i + 1 + (j + 1) * (N + 1)], pnums[i + 1 + j * (N + 1)]]))
#         # netmesh.Add(Element1D([pnums[i + j * (N + 1)], pnums[i + 1 + j * (N + 1)]], index=1))
#         # netmesh.Add(Element1D([pnums[i + j * (N + 1)], pnums[i + (j + 1) * (N + 1)]], index=1))

# #     netmesh.Add(Element1D([pnums[N + j * (N + 1)], pnums[N + (j + 1) * (N + 1)]], index=1))
# #     netmesh.Add(Element1D([pnums[0 + j * (N + 1)], pnums[0 + (j + 1) * (N + 1)]], index=1))

# # for i in range(N):
# #     netmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))
# #     netmesh.Add(Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=1))
# print("a")
# mesh = Mesh(netmesh)
# print("a")
# Vvis = Lagrange(mesh, order=1)
# print("a")
# svis = GridFunction(Vvis)
# print("a")
# svis.vec = s[(N + 1) ** 2:]
# print("a")
# Draw(svis)



ts = [0]
masses = [s.sum()]
q = Queue()

def plot_thread(s):
    fig_sol = plt.figure()

    ax_e = fig_sol.add_subplot(211)
    data_e = s[(N+1) ** 2:].reshape((N+1, N+1))
    im_e = ax_e.imshow(data_e, interpolation='nearest', origin='bottom',
                        aspect='auto', vmin=np.min(data_e), vmax=np.max(data_e))
    plt.colorbar(im_e)

    ax_c = fig_sol.add_subplot(212)
    data_c = s[:(N+1) ** 2].reshape((N+1, N+1))
    im_c = ax_c.imshow(data_c, interpolation='nearest', origin='bottom',
                        aspect='auto', vmin=np.min(data_c), vmax=np.max(data_c))
    plt.colorbar(im_c)

    fig_mass = plt.figure()
    ax_mass = fig_mass.add_subplot(111)
    line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
    ax_mass.legend()
    plt.show(block=False)
    while True:
        try:
            t, s = q.get_nowait()
            ts.append(t)
            masses.append(s.sum())
            im_e.set_data(s[(N+1) ** 2:].reshape((N+1, N+1)))
            im_c.set_data(s[:(N+1) ** 2].reshape((N+1, N+1)))
            line_mass.set_xdata(ts)
            line_mass.set_ydata(masses)
            ax_mass.relim()
            ax_mass.autoscale_view()

            if output:
                np.save(outfile, s)
        except Empty:
            pass
        plt.pause(0.05)

thread = threading.Thread(target=plot_thread, args=(s,))
thread.start()
t = 0.0
while t <= tend:
    print("\n\nt = {:10.2f}".format(t))
    # svis.vec = s[(N + 1) ** 2:]
    # Redraw()


    sold = np.copy(s)
    wnorm = 1e99

    # Newton solver
    while wnorm > 1e-9:
        rhs = np.copy(sold)
        rhs -= B.dot(s)
        As = AApply(s)
        rhs -= As
        Alin = AssembleLinearization(s)

        w = splinalg.spsolve(B + Alin, rhs)
        wnorm = np.linalg.norm(w)
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s += w
        # input("")

    t += dt
    q.put((t, s))

print()
if output:
    outfile.close()
plt.show()
