from netgen.geom2d import SplineGeometry, unit_square
from ngsapps.utils import *

# annulus parameters
Rinner = 30
Router = 90
phi0 = 50
vout = 0.1 # cf. supplementaries Fig S4
# vout = 0 # cf. Raphael mail
v0 = 0.2
smearR = 10
smearphi = 10

Lrect = 120

def annulusInPeriodicSquare(order, maxh):
    geo = SplineGeometry()
    geo.AddCircle((0, 0), Rinner, leftdomain=1, rightdomain=1)
    geo.AddCircle((0, 0), Router, leftdomain=1, rightdomain=1)
    MakePeriodicRectangle(geo, (-Lrect, -Lrect), (Lrect, Lrect))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))

    # local swim speed
    # not sure about v0
    v = AnnulusSpeedCF(Rinner, Router, phi0, vout, v0, smearR, smearphi)

    return mesh, v, v.Dx(), v.Dy()

def annulus(order, maxh):
    geo = SplineGeometry()
    geo.AddCircle((0, 0), Rinner, leftdomain=0, rightdomain=1)
    geo.AddCircle((0, 0), Router, leftdomain=1, rightdomain=0)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order)

    # local swim speed
    # not sure about v0
    v = AnnulusSpeedCF(0, Router, phi0, vout, v0, smearR, smearphi)

    return mesh, v, v.Dx(), v.Dy()

def singleSawtooth(order, maxh):
    # sawtooth parameters
    v0 = 0.2
    vmin = 0.001

    geo = SplineGeometry()
    MakePeriodicRectangle(geo, (0, 0), (300, 10))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))

    smear = 10
    v = IfPos(100-smear-x, v0,
                IfPos(100-x, v0+(x-100+smear)/smear*(vmin-v0),
                    IfPos(200-x, vmin+(x-100)/100*(v0-vmin), v0)))

    vdx = IfPos(100-smear-x, 0,
                IfPos(100-x, (vmin-v0)/smear,
                    IfPos(200-x, (v0-vmin)/100, 0)))

    return mesh, v, vdx, CoefficientFunction(0)
