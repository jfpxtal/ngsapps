from math import sin, cos, sqrt
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

N = 1

# diameter of particles
sigma = 0.01

# diffusion coefficients
# translational
DT = 0.01
# rotational
DR = 3*DT/sigma**2

beta = 1 # 1/(kb*T)

# local swim speed
def v(x, y):
    # return 0
    return 0.1*(1+np.cos(np.sqrt(x*x + y*y)))

# shifted Lennard-Jones potential
def F(x, y):
    return 0

thetas = [0]
xs = [0]
ys = [0]

fig, ax = plt.subplots()
line, = plt.plot(xs, ys)
r = np.arange(-100, 100, 0.2)
meshx, meshy = np.meshgrid(r, r)
plt.contourf(r, r, v(meshx, meshy), alpha=0.5)

plt.show(block=False)

while True:
    thetas.append(thetas[-1] + sqrt(2*DR) * normal())
    xs.append(xs[-1] + v(xs[-1], ys[-1])*cos(thetas[-1]) + beta*DT*F(xs[-1], ys[-1]) + sqrt(2*DT)*normal())
    ys.append(ys[-1] + v(xs[-1], ys[-1])*sin(thetas[-1]) + beta*DT*F(xs[-1], ys[-1]) + sqrt(2*DT)*normal())
    # xs.append(v(xs[-1], ys[-1]) * sqrt(2*DT)*normal())
    # ys.append(v(xs[-1], ys[-1]) * sqrt(2*DT)*normal())

    line.set_data(xs, ys)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()

    input()
