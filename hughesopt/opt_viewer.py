# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:13:16 2017

@author: pietschm
"""
from netgen.geom2d import SplineGeometry
from ngsolve import *
import numpy as np
import time


#from geometries import *
from ngsapps.utils import *
from ngsapps.plotting import *
from limiter import *
from DGForms import *
from rungekutta import *

import pickle

def f(u):
    return 1-u

def fprime(u):
    return -1 + 0*u

order = 1
maxh = 0.1
tau = 0.01
tend = 3 # 3.5
times = np.linspace(0.0,tend,np.ceil(tend/tau)) # FIXME: make tend/tau integer
vtkoutput = False

del1 = 0.1 # Regularization parameters
del2 = 0.1
D = 0.05

Na = 1 # Number of agents
width = 1 # width of conv kernel
alpha = 0.01 # Regularization parameter
cK = 1 # Parameter to control strength of attraction
sigK = 2 # Sigma of exponential conv kernel
vels = 0.8*np.ones((Na,times.size)) # Position of agents

eta = 5 # Penalty parameter

usegeo = "circle"
usegeo = "1d"

plotting = True

radius = 8

if usegeo == "circle":
    geo = SplineGeometry()
    geo.AddCircle ( (0.0, 0.0), r=radius, bc="cyl")
    netgenMesh = geo.GenerateMesh(maxh=maxh)
elif usegeo == "1d":
    netgenMesh = Make1DMesh(-radius, radius, maxh)

mesh = Mesh(netgenMesh)

# finite element space
fes = L2(mesh, order=order, flags={'dgjumps': True})
p1fes = L2(mesh, order=1, flags={'dgjumps': True})
v = fes.TrialFunction()
w = fes.TestFunction()

# Gridfunctions
u = GridFunction(fes)
uu = GridFunction(fes)


# Load data
unpickler = pickle.Unpickler(open ("rhodata_nocontrol.dat", "rb"))
rhouncontrolled = unpickler.load()
del unpickler

##
unpickler = pickle.Unpickler(open ("veldata.dat", "rb"))
vels = unpickler.load()
del unpickler

unpickler = pickle.Unpickler(open ("variancedata.dat", "rb"))
Vs = unpickler.load()
del unpickler

unpickler = pickle.Unpickler(open ("rhodata.dat", "rb"))
rhodata = unpickler.load()
del unpickler

unpickler = pickle.Unpickler(open ("agentsdata.dat", "rb"))
agentsdata = unpickler.load()
del unpickler

#unpickler = pickle.Unpickler(open ("Jdata.dat", "rb"))
#Jopt = unpickler.load()
#del unpickler
# 
 
# Interactive Plot
#from matplotlib.widgets import Slider
#fig_sol, (ax, ax_slider) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[10, 1]})
#
#vplot = Plot(u, ax=ax, mesh=mesh)
#uplot = Plot(uu, ax=ax, mesh=mesh)
#linea, = ax.plot(agentsdata[0], [0.2], marker='o', markersize=15, color="red")
#slider = Slider(ax_slider, "Time", tau, tend)
#
#def update(t):
#    k = int(t/tau)-1
#    u.vec.FV().NumPy()[:] = rhodata[k,:]
#    uu.vec.FV().NumPy()[:] = rhouncontrolled[k,:]
#    linea.set_xdata(agentsdata[k])
#    vplot.Redraw()
#    uplot.Redraw()
#    #plt.pause(0.000001)
#
#slider.on_changed(update)
#plt.show(block=False)

# Create movie
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.animation as manimation

#FFMpegWriter = manimation.writers['ffmpeg']
#metadata = dict(title='Movie Test', artist='Matplotlib',
                #comment='Movie support!')
#writer = FFMpegWriter(fps=15, metadata=metadata)
ax = plt.gca()
#plt.axes([0, 1, 0, 1])
vplot = Plot(u, ax=ax, mesh=mesh,linewidth=4,label='contr')
uplot = Plot(uu, ax=ax, mesh=mesh,linewidth=4,label='uncontr')
linea, = ax.plot(agentsdata[0], [0.4], marker='o', markersize=15, color="red")
#plt.legend()

from matplotlib.pyplot import savefig

for k in range(0,times.size):
    u.vec.FV().NumPy()[:] = rhodata[k,:]
    uu.vec.FV().NumPy()[:] = rhouncontrolled[k,:]
    linea.set_xdata(agentsdata[k])
 #   plt.ylim(0,1)
    vplot.Redraw(autoscale=False)
    uplot.Redraw(autoscale=False)
    plt.ylim([0,1])
    plt.show(block=False)
    plt.pause(0.001)
 #   writer.grab_frame()
    #savefig('png/hughesopt_' + str(k) + '.png')
    
#from subprocess import call
#call("")
