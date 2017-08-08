#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:41:00 2017

@author: jfp
"""

from netgen.meshing import Element0D, Element1D, MeshPoint, Mesh as NetMesh
from netgen.csg import Pnt
from netgen.geom2d import SplineGeometry
from ngsolve import *

def abs(x):
    return IfPos(x, x, -x)

def UpwindFormNonDivergence(fes, beta, v, w, h, n):
    aupw = BilinearForm(fes)
    
    aupw += SymbolicBFI(-beta*grad(v)*w)
    aupw += SymbolicBFI( IfPos(beta*n,beta*n,0)*v*w, BND, skeleton=True)
    aupw += SymbolicBFI(beta*n* (v - v.Other())*0.5*(w + w.Other()), skeleton=True)
    aupw += SymbolicBFI(0.5*abs(beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)
    
    return aupw

def SIPForm(D, eta, fes, v, w, h, n, Dirichlet=True):
        # Diffusion
    asip = BilinearForm(fes)
    asip += SymbolicBFI(D*grad(v)*grad(w))
    asip += SymbolicBFI(-D*0.5*(grad(v)+grad(v.Other())) * n * (w - w.Other()), skeleton=True)
    asip += SymbolicBFI(-D*0.5*(grad(w)+grad(w.Other())) * n * (v - v.Other()), skeleton=True)
    asip += SymbolicBFI(D*eta / h * (v - v.Other()) * (w - w.Other()), skeleton=True)
        
    # SEEMS TO WORK -- WHY 
    if Dirichlet:
        asip += SymbolicBFI(-D*0.5*(grad(v)) * n * (w), BND, skeleton=True) #, definedon=topMat)
        asip += SymbolicBFI(-D*0.5*(grad(w)) * n * (v), BND, skeleton=True) #, definedon=topMat)
        asip += SymbolicBFI(D*eta / h * (v) * w, BND, skeleton=True) #, definedon=topMat)
    
    return asip