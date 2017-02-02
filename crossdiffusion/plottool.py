import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import bisect

# fig_mass, (ax_mass, ax_slider) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[10, 1]})
# line_mass, = ax_mass.plot(ts, masses, 'g', label=r'$\int\;c + e$')

ax_plt = plt.subplot2grid((13, 16), (0, 0), rowspan=10, colspan=16)
ax_order = plt.subplot2grid((13, 16), (10, 0), colspan=16)
ax_maxh = plt.subplot2grid((13, 16), (11, 0), colspan=16)
ax_eta = plt.subplot2grid((13, 16), (12, 2), colspan=14)
ax_conv = plt.subplot2grid((13, 16), (12, 0), colspan=1)
ax_form = plt.subplot2grid((13, 16), (12, 1), colspan=1)

# params = ['order', 'maxh', 'eta']
grid = []
grid.append([1, 2, 3])
grid.append([i*0.02 for i in range(3, 14)])
grid.append([20, 25, 30, 50, 100, 200, 1000])
print(grid)

sliders = []
sliders.append(Slider(ax_order, 'order', 1, 3, 1, valfmt='%0.0f'))
sliders.append(Slider(ax_maxh, 'maxh', 0.06, 0.26, 0.06, valfmt='%0.2f'))
sliders.append(Slider(ax_eta, 'eta', 20, 1000, 20, valfmt='%0.0f'))

conv = False
DG = True
cb_conv = CheckButtons(ax_conv, ['conv'], [conv])
cb_form = CheckButtons(ax_form, ['DG'], [DG])

vals = []
def update_plot(_=None):
    print('hey')
    print(conv)
    global conv, DG, vals
    oldconv = conv
    oldvals = vals
    oldDG = DG
    vals = [s.val for s in sliders]
    # print([bisect.bisect_right(grid[i], vals[i]) for i in range(3)])
    vals = [grid[i][bisect.bisect_right(grid[i], vals[i])-1] for i in range(3)]
    conv = cb_conv.lines[0][0].get_visible()
    print(conv)
    DG = cb_form.lines[0][0].get_visible()
    if conv == oldconv and DG == oldDG and vals == oldvals:
        print('ret')
        return

    fname = '../data/crossdiff/topf/order' + str(vals[0]) + '_maxh' + str(vals[1])
    if DG:
        fname += '_formDG_eta' + str(vals[2])
    else:
        fname += '_formCG'
    fname += '_conv' + str(conv) + '.csv'

    ts = []
    ents = []
    try:
        with open(fname) as f:
            f.readline() # skip first
            for l in f.readlines():
                lvals = [float(v) for v in l.split(',')]
                ts.append(lvals[0])
                ents.append(lvals[1])
        ax_plt.clear()
        ax_plt.plot(ts, ents)
        ax_plt.set_title(fname)
    except FileNotFoundError:
        ax_plt.set_title(fname + ' not found', color='red')
    plt.gcf().canvas.draw()

for s in sliders:
    s.on_changed(update_plot)

cb_conv.on_clicked(update_plot)
cb_form.on_clicked(update_plot)

update_plot()
plt.show()
