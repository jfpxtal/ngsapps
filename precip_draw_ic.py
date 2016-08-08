import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation

import tkinter as tk
from tkinter import filedialog

N = 70
L = 700
alpha = 0.2
span = 3
interpolation = 'nearest'
initialfile = 'precip.ic'

class DrawIC:
    def __init__(self):
        self.span = span
        self.fig_sol, (self.ax_button, self.ax_e, self.ax_c) = \
            plt.subplots(3, 1, gridspec_kw={'height_ratios':[1, 10, 10]})

        self.data_e = np.full((N+1, N+1), alpha)
        self.im_e = self.ax_e.imshow(self.data_e, interpolation=interpolation, origin='bottom',
                                     aspect='auto', vmin=0.0, vmax=1.0, animated=True)
        self.col_e = self.fig_sol.colorbar(self.im_e, ax=self.ax_e, label='e')

        self.data_c = np.zeros((N+1, N+1))
        self.im_c = self.ax_c.imshow(self.data_c, interpolation=interpolation, origin='bottom',
                                     aspect='auto', vmin=-1.0, vmax=1.0, animated=True)
        self.col_c = self.fig_sol.colorbar(self.im_c, ax=self.ax_c, label='c')

        self.ax_button_load = plt.axes([0, 0.95, 0.25, 0.04])
        self.button_load = Button(self.ax_button_load, 'Load from file')
        self.ax_button_save = plt.axes([0.25, 0.95, 0.25, 0.04])
        self.button_save = Button(self.ax_button_save, 'Save to file')
        self.ax_button_reset_e = plt.axes([0.5, 0.95, 0.25, 0.04])
        self.button_reset_e = Button(self.ax_button_reset_e, 'Reset e')
        self.ax_button_reset_c = plt.axes([0.75, 0.95, 0.25, 0.04])
        self.button_reset_c = Button(self.ax_button_reset_c, 'Reset c')


        self.button_load.on_clicked(self.on_load)
        self.button_save.on_clicked(self.on_save)
        self.button_reset_e.on_clicked(self.on_reset_e)
        self.button_reset_c.on_clicked(self.on_reset_c)
        self.fig_sol.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig_sol.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig_sol.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.posx = 0
        self.posy = 0
        self.inaxes = None
        self.pressed = 0
        self.redraw_e = False
        self.redraw_c = False

        self.anim = animation.FuncAnimation(self.fig_sol, self.animate, interval=20, blit=True)
        plt.show()

    def on_press(self, event):
        if event.xdata and event.ydata:
            self.pressed = event.button
            self.posx = (int) (event.xdata + 0.5)
            self.posy = (int) (event.ydata + 0.5)
            self.inaxes = event.inaxes
            # print('PRESSED button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #       (event.button, event.x, event.y, event.xdata, event.ydata))

    def on_motion(self, event):
        if self.pressed and event.xdata and event.ydata:
            self.posx = (int) (event.xdata + 0.5)
            self.posy = (int) (event.ydata + 0.5)
            self.inaxes = event.inaxes
            # print('MOTION x=%d, y=%d, xdata=%f, ydata=%f' %
            #     (event.x, event.y, event.xdata, event.ydata))

    def on_release(self, event):
        self.pressed = 0
        # print('RELEASED button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       (event.button, event.x, event.y, event.xdata, event.ydata))

    def animate(self, i):
        if self.pressed:
            if self.inaxes == self.ax_e:
                box = self.data_e[max(0,self.posy-self.span):min(N,self.posy+self.span)+1,
                            max(0,self.posx-self.span):min(N,self.posx+self.span)+1]
                box += 0.1 * (2 - self.pressed)
                box[box < 0] = 0
                box[box > 1] = 1
                self.redraw_e = True
            elif self.inaxes == self.ax_c:
                box = self.data_c[max(0,self.posy-self.span):min(N,self.posy+self.span)+1,
                            max(0,self.posx-self.span):min(N,self.posx+self.span)+1]
                box += 0.1 * (2 - self.pressed)
                box[box < -1] = -1
                box[box > 1] = 1
                self.redraw_c = True

        artists = []
        if self.redraw_e:
            self.redraw_e = False
            self.im_e.set_data(self.data_e)
            artists.append(self.im_e)
        elif self.redraw_c:
            self.redraw_c = False
            self.im_c.set_data(self.data_c)
            artists.append(self.im_c)

        return artists

    def load_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(initialfile=initialfile)
        root.destroy()

        return file_path

    def save_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(initialfile=initialfile)
        root.destroy()

        return file_path

    def on_load(self, event):
        path = self.load_dialog()
        if path:
            with open(path, "rb") as infile:
                N1 = np.load(infile)
                N2 = np.load(infile)
                if N1 != N or N2 != N:
                    print('Load: number of grid points does not match')
                    return
                Lx = np.load(infile)
                Ly = np.load(infile)
                if Lx != L or Ly != L:
                    print('Load: side length does not match')
                    return
                s = np.load(infile)
            self.data_c = s[:(N + 1) ** 2].reshape((N+1, N+1))
            self.data_e = s[(N + 1) ** 2:].reshape((N+1, N+1))
            self.redraw_c = True
            self.redraw_e = True
            print('Loaded.')

    def on_save(self, event):
        path = self.save_dialog()
        if path:
            with open(path, "wb") as outfile:
                np.save(outfile, N)
                np.save(outfile, N)
                np.save(outfile, L)
                np.save(outfile, L)
                np.save(outfile, np.hstack((self.data_c.flatten(), self.data_e.flatten())))
            print('Saved.')

    def on_reset_e(self, event):
        self.data_e.fill(alpha)
        self.redraw_e = True

    def on_reset_c(self, event):
        self.data_c.fill(0)
        self.redraw_c = True

drawic = DrawIC()
