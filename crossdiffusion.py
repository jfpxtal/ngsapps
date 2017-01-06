from netgen.geom2d import SplineGeometry
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *

order = 3
maxh = 0.15

convOrder = 3;

# diffusion coefficients
# red species
Dr = 0.1
# blue species
Db = 0.3

# advection potentials
# gradVr = CoefficientFunction((1.0, 0.0))
# gradVb = -gradVr
Vr = x
Vb = -x

# time step and end
tau = 0.01
tend = -1

# jump penalty
eta = 100

# geometry and mesh
geo = SplineGeometry()
# set up two rectangles
# the top one is used as domain for the actual calculations and for the visualization of species red
# the bottom one is only used for visualization of species blue
geo.SetMaterial(1, 'top')
geo.SetMaterial(2, 'bottom')
geo.AddRectangle((0, 0), (2, 1), leftdomain=1)
geo.AddRectangle((0, -1.3), (2, -0.3), leftdomain=2)
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
topMat = mesh.Materials('top')

# finite element space
fes1 = L2(mesh, order=order)
# calculations only on top rectangle
fes = FESpace([fes1, fes1], flags={'definedon': ['top'], 'dgjumps': True})

r, b = fes.TrialFunction()
tr, tb = fes.TestFunction()

# initial values
s = GridFunction(fes)
r2 = s.components[0]
b2 = s.components[1]
# r2.Set(IfPos(0.2-x, IfPos(0.5-y, 0.9, 0), 0), definedon=topMat)
# b2.Set(IfPos(x-1.8, 0.6, 0), definedon=topMat)
r2.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)), definedon=topMat)
b2.Set(0.5*exp(-pow(x-1.9, 2)-0.1*pow(y-0.5, 2)), definedon=topMat)

# convolution
convr = Convolve(r2, GaussKernel(scal=20, var=200), mesh, convOrder)
convb = Convolve(b2, GaussKernel(scal=20, var=200), mesh, convOrder)

# GridFunctions for caching of convolution values and automatic gradient calculation
grid = GridFunction(fes)
gridr = grid.components[0]
gridb = grid.components[1]
velocityr = (1-r2-b2)*grad(gridr)
velocityb = (1-r2-b2)*grad(gridb)

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

# special values for DG
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

a = BilinearForm(fes)

# symmetric weighted interior penalty method
# for the diffusion terms

# weights for the averages
# doesn't work, GridFunction doesn't support .Other() ??
# wr = r2*r2.Other() / (r2+r2.Other())
# wb = b2*b2.Other() / (b2+b2.Other())
wr = wb = 0.5

# equation for r
a += SymbolicBFI(Dr*grad(r)*grad(tr))
a += SymbolicBFI(-Dr*0.5*(grad(r) + grad(r.Other())) * n * (tr - tr.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(-Dr*0.5*(grad(tr) + grad(tr.Other())) * n * (r - r.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(Dr*eta / h * (r - r.Other()) * (tr - tr.Other()), skeleton=True, definedon=topMat)

a += SymbolicBFI(-Dr*b2*grad(r)*grad(tr))
a += SymbolicBFI(Dr*wb*(grad(r) + grad(r.Other())) * n * (tr - tr.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(Dr*wb*(grad(tr) + grad(tr.Other())) * n * (r - r.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(-Dr*2*wb*eta / h * (r - r.Other()) * (tr - tr.Other()), skeleton=True, definedon=topMat)

a += SymbolicBFI(Dr*r2*grad(b)*grad(tr))
a += SymbolicBFI(-Dr*wr*(grad(b) + grad(b.Other())) * n * (tr - tr.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(-Dr*wr*(grad(tr)+grad(tr.Other())) * n * (b - b.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(Dr*2*wr*eta / h * (b - b.Other()) * (tr - tr.Other()), skeleton=True, definedon=topMat)

# equation for b
a += SymbolicBFI(Db*grad(b)*grad(tb))
a += SymbolicBFI(-Db*0.5*(grad(b) + grad(b.Other())) * n * (tb - tb.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(-Db*0.5*(grad(tb) + grad(tb.Other())) * n * (b - b.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(Db*eta / h * (b - b.Other()) * (tb - tb.Other()), skeleton=True, definedon=topMat)

a += SymbolicBFI(-Db*r2*grad(b)*grad(tb))
a += SymbolicBFI(Db*wr*(grad(b) + grad(b.Other())) * n * (tb - tb.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(Db*wr*(grad(tb) + grad(tb.Other())) * n * (b - b.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(-Db*2*wr*eta / h * (b - b.Other()) * (tb - tb.Other()), skeleton=True, definedon=topMat)

a += SymbolicBFI(Db*b2*grad(r)*grad(tb))
a += SymbolicBFI(-Db*wb*(grad(r) + grad(r.Other())) * n * (tb - tb.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(-Db*wb*(grad(tb) + grad(tb.Other())) * n * (r - r.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(Db*2*wb*eta / h * (r - r.Other()) * (tb - tb.Other()), skeleton=True, definedon=topMat)

def abs(x):
    return IfPos(x, x, -x)

# upwind scheme for the advection
# missing boundary term??

# equation for r
a += SymbolicBFI(-r*velocityr*grad(tr))
a += SymbolicBFI(velocityr*n*0.5*(r + r.Other())*(tr - tr.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(0.5*abs(velocityr*n) * (r - r.Other())*(tr - tr.Other()), skeleton=True, definedon=topMat)

# equation for b
a += SymbolicBFI(-b*velocityb*grad(tb))
a += SymbolicBFI(velocityb*n*0.5*(b + b.Other())*(tb - tb.Other()), skeleton=True, definedon=topMat)
a += SymbolicBFI(0.5*abs(velocityb*n) * (b - b.Other())*(tb - tb.Other()), skeleton=True, definedon=topMat)

# mass matrix
m = BilinearForm(fes)
m += SymbolicBFI(r*tr)
m += SymbolicBFI(b*tb)

print('Assembling m...')
m.Assemble()

rhs = s.vec.CreateVector()
mstar = m.mat.CreateMatrix()

# Draw(r2, mesh, 'r')
# Draw(b2, mesh, 'b')
# visualize both species at the same time, red in top rectangle, blue in bottom
# translate density b2 of blue species to bottom rectangle
both = r2 + Compose((x, y+1.3), b2, mesh)
Draw(both, mesh, 'both')

times = [0.0]
entropy = ZLogZCF(r2) + ZLogZCF(b2) + ZLogZCF(1-r2-b2) + r2*Vr + b2*Vb
ents = [Integrate(entropy, mesh, definedon=topMat)]
fig, ax = plt.subplots()
line, = ax.plot(times, ents)
plt.show(block=False)

# input("Press any key...")
# semi-implicit Euler
t = 0.0
with TaskManager():
    while tend < 0 or t < tend - tau / 2:
        print("\nt = {:10.6e}".format(t))
        t += tau

        print('Calculating convolution integrals...')
        gridr.Set(Vr+convr, definedon=topMat)
        gridb.Set(Vb+convb, definedon=topMat)
        print('Assembling a...')
        a.Assemble()

        rhs.data = m.mat * s.vec

        mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse(fes.FreeDofs())
        s.vec.data = invmat * rhs

        Redraw(blocking=False)
        times.append(t)
        ents.append(Integrate(entropy, mesh, definedon=topMat))
        line.set_xdata(times)
        line.set_ydata(ents)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        # input()
