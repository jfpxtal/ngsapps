from netgen.geom2d import SplineGeometry, unit_square
from ngsapps.utils import *

# annulus parameters
Rinner = 30
Router = 90
phi0 = 50
vout = 0.1
v0 = 0.2

def annulusInPeriodicSquare(order, maxh):
    geo = SplineGeometry()
    geo.AddCircle((0, 0), Rinner, leftdomain=1, rightdomain=1)
    geo.AddCircle((0, 0), Router, leftdomain=1, rightdomain=1)
    MakePeriodicRectangle(geo, (-100, -100), (100, 100))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))

    # local swim speed
    # not sure about v0
    v = AnnulusSpeedCF(Rinner, Router, phi0, vout, v0)

    return mesh, v, v.Dx(), v.Dy()

def annulus(order, maxh):
    geo = SplineGeometry()
    geo.AddCircle((0, 0), Rinner, leftdomain=0, rightdomain=1)
    geo.AddCircle((0, 0), Router, leftdomain=1, rightdomain=0)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order)

    # local swim speed
    # not sure about v0
    v = AnnulusSpeedCF(0, Router, phi0, vout, v0)

    return mesh, v, v.Dx(), v.Dy()
