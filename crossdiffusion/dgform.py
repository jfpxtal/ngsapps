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
        r, b = self.fes.TrialFunction()
        tr, tb = self.fes.TestFunction()
        r2 = p.s.components[0]
        b2 = p.s.components[1]
        velor = velocities[0]
        velob = velocities[1]

        # special values for DG
        n = specialcf.normal(2)
        h = specialcf.mesh_size

        # symmetric weighted interior penalty method
        # for the diffusion terms

        # weights for the averages
        # doesn't work, GridFunction doesn't support .Other() ??
        # wr = r2*r2.Other() / (r2+r2.Other())
        # wb = b2*b2.Other() / (b2+b2.Other())
        wr = wb = 0.5

        self.a = BilinearForm(self.fes)

        # equation for r
        self.a += SymbolicBFI(p.Dr*grad(r)*grad(tr))
        self.a += SymbolicBFI(-p.Dr*0.5*(grad(r) + grad(r.Other())) * n * (tr - tr.Other()), skeleton=True)
        self.a += SymbolicBFI(-p.Dr*0.5*(grad(tr) + grad(tr.Other())) * n * (r - r.Other()), skeleton=True)
        self.a += SymbolicBFI(p.Dr*self.eta / h * (r - r.Other()) * (tr - tr.Other()), skeleton=True)

        self.a += SymbolicBFI(-p.Dr*b2*grad(r)*grad(tr))
        self.a += SymbolicBFI(p.Dr*wb*(grad(r) + grad(r.Other())) * n * (tr - tr.Other()), skeleton=True)
        self.a += SymbolicBFI(p.Dr*wb*(grad(tr) + grad(tr.Other())) * n * (r - r.Other()), skeleton=True)
        self.a += SymbolicBFI(-p.Dr*2*wb*self.eta / h * (r - r.Other()) * (tr - tr.Other()), skeleton=True)

        self.a += SymbolicBFI(p.Dr*r2*grad(b)*grad(tr))
        self.a += SymbolicBFI(-p.Dr*wr*(grad(b) + grad(b.Other())) * n * (tr - tr.Other()), skeleton=True)
        self.a += SymbolicBFI(-p.Dr*wr*(grad(tr)+grad(tr.Other())) * n * (b - b.Other()), skeleton=True)
        self.a += SymbolicBFI(p.Dr*2*wr*self.eta / h * (b - b.Other()) * (tr - tr.Other()), skeleton=True)

        # equation for b
        self.a += SymbolicBFI(p.Db*grad(b)*grad(tb))
        self.a += SymbolicBFI(-p.Db*0.5*(grad(b) + grad(b.Other())) * n * (tb - tb.Other()), skeleton=True)
        self.a += SymbolicBFI(-p.Db*0.5*(grad(tb) + grad(tb.Other())) * n * (b - b.Other()), skeleton=True)
        self.a += SymbolicBFI(p.Db*self.eta / h * (b - b.Other()) * (tb - tb.Other()), skeleton=True)

        self.a += SymbolicBFI(-p.Db*r2*grad(b)*grad(tb))
        self.a += SymbolicBFI(p.Db*wr*(grad(b) + grad(b.Other())) * n * (tb - tb.Other()), skeleton=True)
        self.a += SymbolicBFI(p.Db*wr*(grad(tb) + grad(tb.Other())) * n * (b - b.Other()), skeleton=True)
        self.a += SymbolicBFI(-p.Db*2*wr*self.eta / h * (b - b.Other()) * (tb - tb.Other()), skeleton=True)

        self.a += SymbolicBFI(p.Db*b2*grad(r)*grad(tb))
        self.a += SymbolicBFI(-p.Db*wb*(grad(r) + grad(r.Other())) * n * (tb - tb.Other()), skeleton=True)
        self.a += SymbolicBFI(-p.Db*wb*(grad(tb) + grad(tb.Other())) * n * (r - r.Other()), skeleton=True)
        self.a += SymbolicBFI(p.Db*2*wb*self.eta / h * (r - r.Other()) * (tb - tb.Other()), skeleton=True)

        def abs(x):
            return IfPos(x, x, -x)

        # upwind scheme for the advection
        # missing boundary term??

        # equation for r
        self.a += SymbolicBFI(-r*(1-r2-b2)*velor*grad(tr))
        self.a += SymbolicBFI((1-r2-b2)*velor*n*0.5*(r + r.Other())*(tr - tr.Other()), skeleton=True)
        self.a += SymbolicBFI(0.5*abs((1-r2-b2)*velor*n) * (r - r.Other())*(tr - tr.Other()), skeleton=True)

        # equation for b
        self.a += SymbolicBFI(-b*(1-r2-b2)*velob*grad(tb))
        self.a += SymbolicBFI((1-r2-b2)*velob*n*0.5*(b + b.Other())*(tb - tb.Other()), skeleton=True)
        self.a += SymbolicBFI(0.5*abs((1-r2-b2)*velob*n) * (b - b.Other())*(tb - tb.Other()), skeleton=True)

        return self.a

    def __str__(self):
        return 'DG_eta' + str(self.eta)
