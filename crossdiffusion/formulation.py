from abc import ABCMeta, abstractmethod

from ngsolve import *

class Formulation(metaclass=ABCMeta):
    @abstractmethod
    def FESpace(self, mesh, order):
        pass

    @abstractmethod
    def BilinearForm(self, params, velocities):
        pass
