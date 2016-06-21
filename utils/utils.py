from libngsapps_utils import *

from ngsolve import *

def Lagrange(mesh, **args):
    """
    Create H1 finite element space with Lagrange basis.
    documentation of arguments is available in FESpace.
    """
    fes = FESpace("lagrangefespace", mesh, **args)
    return fes
