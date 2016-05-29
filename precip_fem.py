# from netgen.geom2d import Element1D, PointId, MeshPoint, Point, Mesh as NgMesh
# from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square
from ngsolve import *

order = 3

tau = 0.05
# tau = 1e-3
tend = 10

gamma = 0.1
alpha = 0.2
kappa = 0

vtkoutput = False

# 1D meshes??!!

# el = Element1D([PointId(0), PointId(1)])
# ngmesh = NgMesh()
# ngmesh.Add(MeshPoint(Point(0,0,0)))
# ngmesh.Add(MeshPoint(Point(1,0,0)))
# ngmesh.Add(PointId(0))
# ngmesh.Add(PointId(1))
# ngmesh.Add(el)
# mesh = Mesh(ngmesh)
# geo = SplineGeometry()
# p1 = geo.AppendPoint(0,0)
# p2 = geo.AppendPoint(0,700)
# geo.Append(["line", p1, p2])
mesh = Mesh(unit_square.GenerateMesh(maxh=0.08))

V = H1(mesh, order=order)
fes = FESpace([V, V])
c, e = fes.TrialFunction()
tc, te = fes.TestFunction()

a = BilinearForm(fes)
a += SymbolicBFI(grad(c) * grad(tc))
a += SymbolicBFI(e * (1 - e) * (e - alpha) * tc)
a += SymbolicBFI(gamma * c * tc)
a += SymbolicBFI(kappa * grad(e) * grad(te))
a += SymbolicBFI(-e * (1 - e) * (e - alpha) * te)
a += SymbolicBFI(-gamma * c * te)

b = BilinearForm(fes)
b += SymbolicBFI(c * tc)
b += SymbolicBFI(e * te)

b.Assemble()

mstar = b.mat.CreateMatrix()

s = GridFunction(fes)

s.components[0].Set(IfPos(1 / 35 - x, 1, 0))
s.components[1].Set(CoefficientFunction(alpha))

rhs = s.vec.CreateVector()
sold = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

Draw(s.components[0], mesh, "c")
Draw(s.components[1], mesh, "e")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[1],s.components[0]],names=["e","c"],filename="precipitation_",subdivision=3)
    vtk.Do()

input("Press any key...")
# implicit Euler
t = 0.0
while t < tend:
    print("\n\nt = {:10.6e}".format(t))

    sold.data = s.vec
    wnorm = 1e99

    # newton solver
    while wnorm > 1e-9:
        rhs.data = b.mat * sold
        rhs.data -= b.mat * s.vec
        a.Apply(s.vec,As)
        rhs.data -= tau * As
        a.AssembleLinearization(s.vec)

        mstar.AsVector().data = b.mat.AsVector() + tau * a.mat.AsVector()
        invmat = mstar.Inverse()
        w.data = invmat * rhs
        wnorm = w.Norm()
        print("|w| = {:7.3e} ".format(wnorm),end="")
        s.vec.data += w

    t += tau
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
