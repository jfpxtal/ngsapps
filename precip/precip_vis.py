import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

f = open("precip.bin", "rb")

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

L = 700
dt = 0.05
its = 100
tend = dt * its * len(ss)
N = int(len(ss[0]) / 2 - 1)

xs = np.linspace(0, L, num=N + 1)
ts = np.linspace(0, tend, num=len(masses))

fig_sol, (ax_e, ax_c, ax_slider) = plt.subplots(3, 1, gridspec_kw={'height_ratios':[10, 10, 1]})

line_e, = ax_e.plot(xs, ss[0][N + 1:], "b", label="e")
ax_e.legend()

line_c, = ax_c.plot(xs, ss[0][:N + 1], "b", label="c")
ax_c.legend()

fig_mass = plt.figure()
ax_mass = fig_mass.add_subplot(111)
line_mass, = ax_mass.plot(ts, masses, "g", label=r"$\int\;c + e$")
ax_mass.legend(loc=2)

fig_heat = plt.figure()
ax_heat = fig_heat.add_subplot(111)
data = np.expand_dims(ss[0][N + 1:], axis=0)
im = ax_heat.imshow(data, interpolation='nearest', origin='bottom',
                    aspect='auto', vmin=np.min(data), vmax=np.max(data))

slider = Slider(ax_slider, "Time", 0, tend - 1)

tline = ax_mass.axvline(0, color='r')

def update(t):
    tline.set_xdata(t)
    t = int(t / (dt * its))
    line_e.set_ydata(ss[t][N + 1:])
    line_c.set_ydata(ss[t][:N + 1])
    ax_e.relim()
    ax_e.autoscale_view()
    ax_c.relim()
    ax_c.autoscale_view()
    fig_sol.canvas.draw_idle()
    fig_mass.canvas.draw_idle()
    im.set_data(np.expand_dims(ss[t][N + 1:], axis=0))
    fig_heat.canvas.draw_idle()

slider.on_changed(update)

plt.show()
