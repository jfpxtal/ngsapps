# Solve cross-diffusion system
# r_t = div( (1-rho)grad r + r grad rho + r(1-rho)grad V )
# b_t = div( (1-rho)grad b + b grad rho + b(1-rho)grad W )
# with rho = r + b
from netgen.geom2d import unit_square
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *

order = 3
maxh = 0.15

# time step and end
tau = 0.05
tend = 25

ngsglobals.msg_level = 1

# diffusion coefficients
# red species
Dr = 0.1
# blue species
Db = 0.3


#mesh = Mesh (unit_square.GenerateMesh(maxh=0.1))
from netgen.geom2d import SplineGeometry
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


# H1-conforming finite element space
fes1 = H1(mesh, order=order) # Neumann only, dirichlet=[1,2,3,4])
fes = FESpace([fes1,fes1])

r,b = fes.TrialFunction()
tr,tb = fes.TestFunction()

# initial values
s = GridFunction(fes)
r2 = s.components[0]
b2 = s.components[1]
# r2.Set(IfPos(0.2-x, IfPos(0.5-y, 0.9, 0), 0), definedon=topMat)
# b2.Set(IfPos(x-1.8, 0.6, 0), definedon=topMat)
r2.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)), definedon=topMat)
b2.Set(0.5*exp(-pow(x-1.9, 2)-0.1*pow(y-0.5, 2)), definedon=topMat)

u = GridFunction (fes)
rhs = u.vec.CreateVector()
uold = u.vec.CreateVector()

potV = GridFunction(fes1)
potW = GridFunction(fes1)
potV.Set(0*x) #CoefficientFunction( (1,0) ))
#potV = CoefficientFunction( (1,0) ) #y-0.5,0.5-x) )
#potW = CoefficientFunction( (-1,0) ) #y-0.5,0.5-x) )
potW.Set(0*x) #CoefficientFunction (-1,0) )
# the right hand side
#f = LinearForm (fes)
#f += Source (32 * (y*(1-y)+x*(1-x)))

# Flow boundary conditions
alpha1 = 0.7
alpha2 = 0.7
beta1 = 0.6
beta2 = 0.6
eps = 0.1

# the bilinear-form
rho = u.components[0] + u.components[1]
a = BilinearForm (fes, symmetric=False)
a += SymbolicBFI ( Dr*(1-b2)*grad(r)*grad(tr) + Dr*r2*grad(b)*grad(tr) + r*(1-r2-b2)*grad(potV)*grad(tr) )
a += SymbolicBFI ( Db*(1-r2)*grad(b)*grad(tb) + Db*b2*grad(r)*grad(tb) + b*(1-r2-b2)*grad(potW)*grad(tb) )
#a += SymbolicBFI ( Dr*(1-rho)*grad(r)*grad(tr) + eps*u.components[0]*(grad(r)+grad(b))*grad(tr) + r*(1-rho)*grad(potV)*grad(tr) )
#a += SymbolicBFI ( Db*(1-rho)*grad(b)*grad(tb) + eps*u.components[1]*(grad(r)+grad(b))*grad(tb) + b*(1-rho)*grad(potW)*grad(tb) )
#a += SymbolicBFI ( alpha1*(r+b)*tr + alpha2*(r+b)*tb,BND,definedon=[1] )
#a += SymbolicBFI ( beta1*r*tr + beta2*b*tb,BND,definedon=[3] )
m = BilinearForm(fes)
m += SymbolicBFI ( r*tr + b*tb ) #Mass(1)

f = LinearForm(fes)
#f += SymbolicLFI ( alpha1*tr + alpha2*tb,BND,definedon=[1] )
f.Assemble()

m.Assemble()
mmat = m.mat
smat = mmat.CreateMatrix()
#f.Assemble()
#pi = 3.14159

# visualize both species at the same time, red in top rectangle, blue in bottom
# translate density b2 of blue species to bottom rectangle
both = r2 + Compose((x, y+1.3), b2, mesh)
Draw(both, mesh, 'both')

# Calculate constant equilibria
domainSize = Integrate(CoefficientFunction(1),mesh,definedon=topMat)
rinfty = Integrate(r2,mesh, definedon=topMat) / domainSize
binfty = Integrate(b2,mesh, definedon=topMat) / domainSize 

times = [0.0]
entropy = ZLogZCF(r2/rinfty) + ZLogZCF(b2/binfty) + ZLogZCF((1-r2-b2)/(1-rinfty-binfty)) + r2*potV + b2*potW
ents = [Integrate(entropy, mesh, definedon=topMat)]
fig, ax = plt.subplots()
line, = ax.plot(times, ents)
plt.show(block=False)

input("")
t = 0.0
while t < tend:
    a.Assemble()
    smat.AsVector().data = tau * a.mat.AsVector() + mmat.AsVector()
    rhs.data = mmat * s.vec + tau*f.vec
    s.vec.data = smat.Inverse(fes.FreeDofs()) * rhs
#    print(u.components[0].vec)
    t += tau
    print("\nt = {:10.6e}".format(t))
    Redraw(blocking=False)
    times.append(t)
    ents.append(Integrate(entropy, mesh, definedon=topMat))
    line.set_xdata(times)
    line.set_ydata(ents)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()

outfile = open('ents_cg.csv','w')
for item in ents:
        outfile.write("%s\n" % item)
outfile.close()

