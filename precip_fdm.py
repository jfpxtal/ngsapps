# http://www.math.umn.edu/~scheel/preprints/pf0.pdf

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

# L = 10
# N = 10
L = 700
N = 7000
dx = L / N

dt = 0.05
# tend = 4000
tend = -1

gamma = 0.1
alpha = 0.2
kappa = 0

delta = 0.1

outfile = None
# outfile = open("precip.bin", "wb")

# kappa == 0
diag = np.hstack((dt / (dx ** 2) + 1 + dt * gamma, np.full(N - 1, 1 + 2 * dt / (dx ** 2) + dt * gamma), dt / (dx ** 2) + 1 + dt * gamma,
                  1 + kappa * dt / (dx ** 2), np.full(N - 1, 1 + 2 * kappa * dt / (dx ** 2)), 1 + kappa * dt / (dx ** 2)))
upper_minor = np.hstack((0, np.full(N, -dt / (dx ** 2)), 0, np.full(N, -kappa * dt / (dx ** 2))))
lower_minor = np.hstack((np.full(N, -dt / (dx ** 2)), 0, np.full(N, -kappa * dt / (dx ** 2)), 0))
B = sp.dia_matrix((np.vstack((lower_minor, diag, upper_minor)), [-1, 0, 1]), (2 * N + 2, 2 * N + 2))

cdiag = np.full(N + 1, -gamma * dt)
cdia_mat = sp.dia_matrix((cdiag, 0), (N + 1, N + 1))
C = sp.vstack((sp.csr_matrix((N + 1, 2 * N + 2)), sp.hstack((cdia_mat, sp.csc_matrix((N + 1, N + 1))))))

M = B + C
print(M.toarray())

def AApply(u):
    v = u[N + 1:]
    w = v * (1 - v) * (v - alpha)
    # print(dt * np.hstack((w, -w)))
    return (dt * np.hstack((w, -w)))

def AssembleLinearization(u):
    rightm = sp.dia_matrix((-3 * u[N + 1:] ** 2 + 2 * (1 + alpha) * u[N + 1:] - alpha, 0), (N + 1, N + 1))
    Alin = sp.bmat([[sp.coo_matrix((N + 1, N + 1)), rightm], [None, -rightm]])
    # print(dt * Alin.toarray())
    return dt * Alin


s = np.hstack((np.full(10 / dx, delta), np.full(10 / dx, -delta), np.zeros(N + 1 - 20 / dx), np.full(N + 1, alpha)))
# s = np.hstack((np.full(1 / dx, delta), np.full(1 / dx, -delta), np.zeros(N + 1 - 2 / dx), np.full(N + 1, alpha)))


xs = np.linspace(0, L, num=N + 1)
ts = np.array([0])

fig_sol = plt.figure()

ax_e = fig_sol.add_subplot(211)
line_e, = ax_e.plot(xs, s[N + 1:], "b", label="e")
ax_e.legend()

ax_c = fig_sol.add_subplot(212)
line_c, = ax_c.plot(xs, s[:N + 1], "b", label="c")
ax_c.legend()

fig_mass = plt.figure()
ax_mass = fig_mass.add_subplot(111)
masses = np.array([s.sum()])
line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
ax_mass.legend()

plt.show(block=False)

input("Press any key...")
# implicit Euler
t = 0.0
it = 1
while tend < 0 or t < tend - dt / 2:
    print("\n\nt = {:10.2f}".format(t))
    # print(s)
    # if it % 200 == 0:
    if it % 100 == 0:
        # input("")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("mass = " + str(s.sum()))
        line_e.set_ydata(s[N + 1:])
        line_c.set_ydata(s[:N + 1])
        ax_e.relim()
        ax_e.autoscale_view()
        ax_c.relim()
        ax_c.autoscale_view()
        ts = np.append(ts, t)
        masses = np.append(masses, s.sum())
        line_mass.set_xdata(ts)
        line_mass.set_ydata(masses)
        ax_mass.relim()
        ax_mass.autoscale_view()
        fig_sol.canvas.draw()
        fig_mass.canvas.draw()
        if outfile:
            np.save(outfile, s)

    sold = np.copy(s)
    wnorm = 1e99

    # Newton solver
    while wnorm > 1e-9:
        rhs = np.copy(sold)
        rhs -= M.dot(s)
        As = AApply(s)
        rhs -= As
        Alin = AssembleLinearization(s)

        # print("M + Alin = " + str((M + Alin).toarray()))
        w = splinalg.spsolve(M + Alin, rhs)
        # print("w = " + str(w))
        # w = splinalg.spsolve(M, rhs)
        wnorm = np.linalg.norm(w)
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s += w
        # input("")

    t += dt
    it += 1

print()
if outfile:
    outfile.close()
plt.show()
