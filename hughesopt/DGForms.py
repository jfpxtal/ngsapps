#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:41:00 2017

@author: jfp
"""

from ngsolve import *
from ngsapps.utils import *

def UpwindFormNonDivergence(fes, beta, v, w, h, n, Compile=False):
    aupw = BilinearForm(fes)
    
    if Compile:
        aupw += SymbolicBFI((beta*grad(v)*w).Compile())
    else:
        aupw += SymbolicBFI(beta*grad(v)*w)
    aupw += SymbolicBFI(negPart(beta*n)*v*w, BND, skeleton=True)
    aupw += SymbolicBFI(-beta*n*(v - v.Other())*0.5*(w + w.Other()), skeleton=True)
    aupw += SymbolicBFI(0.5*abs(beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)    
    
    return aupw

def UpwindFormDivergence(fes, beta, v, w, h, n):
    aupw = BilinearForm(fes)
    
    aupw += SymbolicBFI((-v*beta*grad(w)).Compile())
    aupw += SymbolicBFI(posPart(beta*n)*v*w, BND, skeleton=True)
    aupw += SymbolicBFI(beta*n*(v + v.Other())*0.5*(w - w.Other()), skeleton=True)
    aupw += SymbolicBFI(0.5*abs(beta*n)*(v - v.Other())*(w - w.Other()), skeleton=True)
    
    return aupw


def SIPForm(D, eta, fes, v, w, h, n, Dirichlet=True):
        # Diffusion
    asip = BilinearForm(fes)
    asip += SymbolicBFI((D*grad(v)*grad(w)).Compile())
    asip += SymbolicBFI(-D*0.5*(grad(v)+grad(v.Other())) * n * (w - w.Other()), skeleton=True)
    asip += SymbolicBFI(-D*0.5*(grad(w)+grad(w.Other())) * n * (v - v.Other()), skeleton=True)
    asip += SymbolicBFI(D*eta / h * (v - v.Other()) * (w - w.Other()), skeleton=True)
        
    # SEEMS TO WORK -- WHY 
    if Dirichlet:
        asip += SymbolicBFI(-D*0.5*(grad(v)) * n * (w), BND, skeleton=True) #, definedon=topMat)
        asip += SymbolicBFI(-D*0.5*(grad(w)) * n * (v), BND, skeleton=True) #, definedon=topMat)
        asip += SymbolicBFI(D*eta / h * (v) * w, BND, skeleton=True) #, definedon=topMat)
    
    return asip
