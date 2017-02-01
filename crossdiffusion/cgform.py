from formulation import *

class CGFormulation(Formulation):
    def FESpace(self, mesh, order):
        # finite element space
        self.fes1 = H1(mesh, order=order, flags={'definedon': ['top']})
        # calculations only on top mesh
        self.fes = FESpace([self.fes1, self.fes1], flags={'definedon': ['top']})
        return self.fes1, self.fes

    def BilinearForm(self, p, velocities):
        r, b = self.fes.TrialFunction()
        tr, tb = self.fes.TestFunction()
        r2 = p.s.components[0]
        b2 = p.s.components[1]
        velor = velocities[0]
        velob = velocities[1]

        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += SymbolicBFI(p.Dr*(1-b2)*grad(r)*grad(tr) + p.Dr*r2*grad(b)*grad(tr) - r*(1-r2-b2)*velor*grad(tr))
        self.a += SymbolicBFI(p.Db*(1-r2)*grad(b)*grad(tb) + p.Db*b2*grad(r)*grad(tb) - b*(1-r2-b2)*velob*grad(tb))
        return self.a
