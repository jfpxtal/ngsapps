# http://www.math.umn.edu/~scheel/preprints/pf0.pdf

import numpy as np
import numpy.random as random
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import multiprocessing as mp

from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from ngsapps.utils import *
from ngsolve import *

np.set_printoptions(linewidth=400, threshold=100000)

# L = 700
Lx = 400
Ly = 200
N = 80
M = 40

dt = 0.2
tend = -1
# tend = 400

gamma = 0.1
alpha = 0.2
kappa = 0

delta = 0.1

continuous_ngplot = True

# load initial conditions from file (see precip_draw_ic.py)
icfile = None
# icfile = open('precip.ic', 'rb')

outfile = None
# outfile = open('precip2d.bin', 'wb')

if icfile:
    N = np.load(icfile)
    M = np.load(icfile)
    Lx = np.load(icfile)
    Ly = np.load(icfile)

dx = Lx / N
dy = Ly / M
total_points = (N + 1) * (M + 1)

rowidxs = []
colidxs = []
data = []

def to_vec_index(func, coord):
    # c00, c10, c20, ... , cN0 , c01, ... , cNM, e00, ...
    i, j = coord
    if func == 'c':
        return i + j * (N + 1)
    elif func == 'e':
        return total_points + i + j * (N + 1)

def newentry(i, j, d):
    # B[i, j] = d
    rowidxs.append(i)
    colidxs.append(j)
    data.append(d)

def add_laplace(func, scale, a, b, c):
    newentry(to_vec_index(func, b), to_vec_index(func, a), scale)
    newentry(to_vec_index(func, b), to_vec_index(func, b), -2 * scale)
    newentry(to_vec_index(func, b), to_vec_index(func, c), scale)

for i in range(N + 1):
    for j in range(M + 1):
        if i > 0 and i < N:
            add_laplace('c', 1 / dx ** 2, (i-1,j), (i,j), (i+1,j))
            add_laplace('e', kappa / dx ** 2, (i-1,j), (i,j), (i+1,j))
        if j > 0 and j < M:
            add_laplace('c', 1 / dy ** 2, (i,j-1), (i,j), (i,j+1))
            add_laplace('e', kappa / dy ** 2, (i,j-1), (i,j), (i,j+1))

        newentry(to_vec_index('c', (i,j)), to_vec_index('c', (i,j)), -gamma)
        newentry(to_vec_index('e', (i,j)), to_vec_index('c', (i,j)), gamma)

def add_neumann(func, scale, bnd, inner):
    newentry(to_vec_index(func, bnd), to_vec_index(func, bnd), -scale)
    newentry(to_vec_index(func, bnd), to_vec_index(func, inner), scale)

for f, s in [('c', 1), ('e', kappa)]:
    for i in range(N + 1):
        add_neumann(f, s / dy ** 2, (i,0), (i,1))
        add_neumann(f, s / dy ** 2, (i,M), (i,M-1))
    for j in range(M + 1):
        add_neumann(f, s / dx ** 2, (0,j), (1,j))
        add_neumann(f, s / dx ** 2, (N,j), (N-1,j))

B = sp.coo_matrix((data, (rowidxs, colidxs)),
                  shape=(2 * total_points, 2 * total_points))
B *= -dt
B += sp.eye(2 * total_points)
B = B.tocsr()
# print(B.todense())


def AApply(u):
    v = u[total_points:]
    w = v * (1 - v) * (v - alpha)
    # print(dt * np.hstack((w, -w)))
    return dt * np.hstack((w, -w))

def AssembleLinearization(u):
    rightm = sp.dia_matrix((-3 * u[total_points:] ** 2
                            + 2 * (1 + alpha) * u[total_points:]
                            - alpha, 0), (total_points, total_points))
    Alin = sp.bmat([[sp.coo_matrix((total_points, total_points)), rightm],
                    [None, -rightm]])
    # print(dt * Alin.toarray())
    return dt * Alin

if continuous_ngplot:
    mesh = Mesh(GenerateGridMesh((0,0), (Lx,Ly), N, M))
    Vvis = H1(mesh, order=1)
else:
    mesh = Mesh(GenerateGridMesh((0,0), (Lx,Ly), N+1, M+1))
    Vvis = L2(mesh, order=0)

fes = FESpace([Vvis, Vvis])
svis = GridFunction(fes)

s = np.zeros(2 * total_points)
if icfile:
    s += np.load(icfile)
    icfile.close()
    svis.vec.FV().NumPy()[:] = s
else:
    # s = random.rand(2 * total_points)
    # s = np.hstack((np.full(2 * (N + 1), delta),
    #                 np.full(2 * (N + 1), -delta),
    #                 np.zeros(total_points - 4 * (N + 1)),
    #                 np.full(total_points, alpha)))
    # s = np.hstack((np.full(10 * (N + 1), delta),
    #                 np.full(10 * (N + 1), -delta),
    #                 np.zeros(total_points - 20 * (N + 1)),
    #                 np.full(total_points, alpha)))
    # svis.vec.FV().NumPy()[:] = s
    width = 200
    svis.components[0].Set(exp(-((x-Lx/2) * (x-Lx/2) + (y-Ly/2) * (y-Ly/2)) / width))
    svis.components[1].Set(CoefficientFunction(alpha))
    s = svis.vec.FV().NumPy()

Draw(svis.components[0], mesh, 'c')
Draw(svis.components[1], mesh, 'e')

def plot_proc(sinit, t_sh, s_sh):
    import matplotlib.pyplot as plt
    ts = [0]
    masses = [sinit.sum()]
    fig_mass = plt.figure()
    ax_mass = fig_mass.add_subplot(111)
    line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
    ax_mass.legend()

    plt.show(block=False)
    while True:
        with t_sh.get_lock(), s_sh.get_lock():
            t = t_sh.value
            s = np.array(s_sh[:])
        if t == -1:
            break
        elif t != ts[-1]:
            ts.append(t)
            masses.append(s.sum())
            line_mass.set_xdata(ts)
            line_mass.set_ydata(masses)
            ax_mass.relim()
            ax_mass.autoscale_view()

        plt.pause(0.05)

    plt.show()

t_sh = mp.Value('d', 0.0)
s_sh = mp.Array('d', s)
proc = mp.Process(target=plot_proc, args=(s, t_sh, s_sh))
proc.start()

input('Press any key...\n\n')
if outfile:
    np.save(outfile, N)
    np.save(outfile, M)
    np.save(outfile, Lx)
    np.save(outfile, Ly)
    np.save(outfile, dt)
    np.save(outfile, s)

# implicit Euler
t = 0.0
while tend < 0 or t < tend - dt / 2:
    print('\n\nt = {:10.2f}'.format(t))

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
        print('|w| = {:7.3e} '.format(wnorm),end='')
        s += w
        # input('')

    t += dt
    with t_sh.get_lock(), s_sh.get_lock():
        t_sh.value = t
        s_sh[:] = s
    svis.vec.FV().NumPy()[:] = s
    Redraw(blocking=False)
    if outfile:
        np.save(outfile, s)

print('\n\nt = {:10.2f}'.format(t))
if outfile:
    outfile.close()
t_sh.value = -1
