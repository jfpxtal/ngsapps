import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)
outfile = open("precip.bin", "wb")

# L = 10
# N = 10
L = 700
N = 7000
dx = L / N

dt = 0.05
# tend = 4000
tend = 1000

gamma = 0.1
alpha = 0.2
kappa = 0

delta = 0.1

diag = np.hstack((1, np.full(N - 1, dx ** 2 / dt + 2 + gamma), 1,
             1, np.full(N - 1, dx ** 2 / dt + 2 * kappa), 1))
upper_minor = np.hstack((0, -np.ones(N), np.full(N + 1, -kappa)))
lower_minor = np.hstack((-np.ones(N), np.full(N, -kappa), -1, 0))
B = sp.dia_matrix((np.vstack((lower_minor, diag, upper_minor)), [-1, 0, 1]), (2 * N + 2, 2 * N + 2))
B *= dt

cdiag = np.hstack((0, np.full(N - 1, -gamma * dx ** 2 * dt), 0))
cdia_mat = sp.dia_matrix((cdiag, 0), (N + 1, N + 1))
C = sp.vstack((sp.csr_matrix((N + 1, 2 * N + 2)), sp.hstack((cdia_mat, sp.csc_matrix((N + 1, N + 1))))))

M = B + C
print(M.toarray())

def AApply(u):
    v = u[N + 2:-1]
    w = v * (1 - v) * (v - alpha)
    return dx ** 2 * dt * np.hstack((0, w, 0, 0, -w, 0))

def AssembleLinearization(u):
    rightm = sp.dia_matrix((np.hstack((0, -3 * u[N + 2:-1] ** 2 + 2 * (1 + alpha) * u[N + 2:-1] - alpha, 0)), 0), (N + 1, N + 1))
    Alin = sp.bmat([[sp.coo_matrix((N + 1, N + 1)), rightm], [None, -rightm]])
    return dx ** 2 * dt * Alin

s = np.hstack((np.full(10 / dx, 20), np.full(10 / dx, -10), np.zeros(N + 1 - 20 / dx), np.full(N + 1, alpha)))

fig = plt.figure()
xs = np.linspace(0, L, num=N + 1)

input("Press any key...")
# implicit Euler
t = 0.0
it = 0
while t < tend:
    it += 1
    print("\n\nt = {:10.6e}".format(t))
    if it % 200 == 0:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        fig.clear()
        plt.plot(xs, s[N + 1:])
        # plt.plot(xs, s[:N+1])
        plt.pause(0.05)
        # plt.show(block=False)
        np.save(outfile, s)

    sold = np.copy(s)
    wnorm = 1e99

    rhs1 = dx ** 2 * np.hstack((0, sold[1:N], 0, 0, sold[N + 2:-1], 0))
    # newton solver
    while wnorm > 1e-9:
        rhs = np.copy(rhs1)
        rhs -= M.dot(s)
        As = AApply(s)
        rhs -= As
        Alin = AssembleLinearization(s)

        w = splinalg.spsolve(M + Alin, rhs)
        wnorm = np.linalg.norm(w)
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s += w

    t += dt

outfile.close()
plt.show()
