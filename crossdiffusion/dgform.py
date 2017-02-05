from formulation import *

class DGFormulation(Formulation):
    def __init__(self, eta):
        self.eta = eta

    def FESpace(self, mesh, order):
        # finite element space
        self.fes1 = L2(mesh, order=order, flags={'definedon': ['top']})
        # calculations only on top mesh
        self.fes = FESpace([self.fes1, self.fes1], flags={'definedon': ['top'], 'dgjumps': True})
        return self.fes1, self.fes

    def BilinearForm(self, p, velocities):
        trial = self.fes.TrialFunction()
        test = self.fes.TestFunction()
        r2 = p.s.components[0]
        b2 = p.s.components[1]
        rho2 = r2+b2
        Ds = [p.Dr, p.Db]

        # special values for DG
        self.n = specialcf.normal(2)
        self.h = specialcf.mesh_size

        self.a = BilinearForm(self.fes)

        for i in range(2):
            self.addSIP(Ds[i] * (1-p.s.components[1-i]), trial[i], test[i])
            self.addSIP(Ds[i] * p.s.components[i], trial[1-i], test[i])

            self.addUpwind((1-rho2) * velocities[i], trial[i], test[i])

        return self.a

    # symmetric interior penalty method
    # for the diffusion terms
    def addSIP(self, coeff, trial, test):
        self.a += SymbolicBFI(coeff * grad(trial) * grad(test))
        self.a += SymbolicBFI(
            coeff * self.eta / self.h * (trial - trial.Other()) * (test - test.Other())
            - coeff * 0.5 * (grad(trial) + grad(trial.Other())) * self.n * (test - test.Other())
            - coeff * 0.5 * (grad(test) + grad(test.Other())) * self.n * (trial - trial.Other()),
            skeleton=True)

    def abs(self, x):
        return IfPos(x, x, -x)

    def posPart(self, x):
        return IfPos(x, x, 0)

    # upwind scheme for advection
    def addUpwind(self, adv, trial, test):
        self.a += SymbolicBFI(-trial * adv * grad(test))
        self.a += SymbolicBFI(
            adv * self.n * 0.5 * (trial + trial.Other()) * (test - test.Other())
            + 0.5 * self.abs(adv * self.n) * (trial - trial.Other()) * (test - test.Other()),
            skeleton=True)

        self.a += SymbolicBFI(self.posPart(adv * self.n) * trial * test, BND, skeleton=True)

    def __str__(self):
        return 'DG_eta' + str(self.eta)
