from netgen.geom2d import SplineGeometry
from ngsolve import *
import matplotlib.pyplot as plt
from ngsapps.utils import *
import numpy as np

for order in [3, 2, 1]:
    for maxh in [0.1, 0.15, 0.3]:
        for eta in [20, 50, 80, 110, 1000]:

            # order = 3
            # maxh = 0.10
            # maxh = 0.1

            convOrder = 3

            # diffusion coefficients
            # red species
            Dr = 0.1
            # blue species
            Db = 0.3

            # advection potentials
            # gradVr = CoefficientFunction((1.0, 0.0))
            # gradVb = -gradVr
            Vr = -x
            Vb = x

            # time step and end
            tau = 0.05
            tend = 8

            # jump penalty
            # eta = 20

            # geometry and mesh
            geo = SplineGeometry()
            # set up two rectangles
            # the top one is used as domain for the actual calculations and for the visualization of species red
            # the bottom one is only used for visualization of species blue
            geo.SetMaterial(1, 'top')
            geo.SetMaterial(2, 'bottom')
            geo.AddRectangle((0, 0), (2, 1), leftdomain=1)
            geo.AddRectangle((0, -1.3), (2, -0.3), leftdomain=2)
            mesh = Mesh(geo.GenerateMesh(maxh=maxh))
            topMat = mesh.Materials('top')

            # finite element space
            fes1 = L2(mesh, order=order, flags={'definedon': ['top']})
            # calculations only on top rectangle
            fes = FESpace([fes1, fes1], flags={'definedon': ['top'], 'dgjumps': True})

            r, b = fes.TrialFunction()
            tr, tb = fes.TestFunction()

            # initial values
            s = GridFunction(fes)
            r2 = s.components[0]
            b2 = s.components[1]
            # r2.Set(IfPos(0.2-x, IfPos(0.5-y, 0.9, 0), 0))
            # b2.Set(IfPos(x-1.8, 0.6, 0))
            r2.Set(0.5*exp(-pow(x-0.1, 2)-pow(y-0.25, 2)))
            b2.Set(0.5*exp(-pow(x-1.9, 2)-0.1*pow(y-0.5, 2)))
            #r2.Set(0.5+0*x)
            #b2.Set(0.5+0*x)

            # convolution
            thin = 200
            k0 = 20
            K = k0*exp(-thin*(x*x+y*y))
            convr = Convolve(r2, K, mesh, convOrder)
            convb = Convolve(b2, K, mesh, convOrder)

            # GridFunctions for caching of convolution values and automatic gradient calculation
            grid = GridFunction(fes)
            gridr = grid.components[0]
            gridb = grid.components[1]
            gridr.Set(Vr+0*convr)
            gridb.Set(Vb+0*convb)
            velocityr = -(1-r2-b2)*grad(gridr)
            velocityb = -(1-r2-b2)*grad(gridb)

            # special values for DG
            n = specialcf.normal(mesh.dim)
            h = specialcf.mesh_size

            a = BilinearForm(fes)

            # symmetric weighted interior penalty method
            # for the diffusion terms

            # weights for the averages
            # doesn't work, GridFunction doesn't support .Other() ??
            # wr = r2*r2.Other() / (r2+r2.Other())
            # wb = b2*b2.Other() / (b2+b2.Other())
            wr = wb = 0.5

            # equation for r
            a += SymbolicBFI(Dr*grad(r)*grad(tr))
            a += SymbolicBFI(-Dr*0.5*(grad(r) + grad(r.Other())) * n * (tr - tr.Other()), skeleton=True)
            a += SymbolicBFI(-Dr*0.5*(grad(tr) + grad(tr.Other())) * n * (r - r.Other()), skeleton=True)
            a += SymbolicBFI(Dr*eta / h * (r - r.Other()) * (tr - tr.Other()), skeleton=True)

            a += SymbolicBFI(-Dr*b2*grad(r)*grad(tr))
            a += SymbolicBFI(Dr*wb*(grad(r) + grad(r.Other())) * n * (tr - tr.Other()), skeleton=True)
            a += SymbolicBFI(Dr*wb*(grad(tr) + grad(tr.Other())) * n * (r - r.Other()), skeleton=True)
            a += SymbolicBFI(-Dr*2*wb*eta / h * (r - r.Other()) * (tr - tr.Other()), skeleton=True)

            a += SymbolicBFI(Dr*r2*grad(b)*grad(tr))
            a += SymbolicBFI(-Dr*wr*(grad(b) + grad(b.Other())) * n * (tr - tr.Other()), skeleton=True)
            a += SymbolicBFI(-Dr*wr*(grad(tr)+grad(tr.Other())) * n * (b - b.Other()), skeleton=True)
            a += SymbolicBFI(Dr*2*wr*eta / h * (b - b.Other()) * (tr - tr.Other()), skeleton=True)

            # equation for b
            a += SymbolicBFI(Db*grad(b)*grad(tb))
            a += SymbolicBFI(-Db*0.5*(grad(b) + grad(b.Other())) * n * (tb - tb.Other()), skeleton=True)
            a += SymbolicBFI(-Db*0.5*(grad(tb) + grad(tb.Other())) * n * (b - b.Other()), skeleton=True)
            a += SymbolicBFI(Db*eta / h * (b - b.Other()) * (tb - tb.Other()), skeleton=True)

            a += SymbolicBFI(-Db*r2*grad(b)*grad(tb))
            a += SymbolicBFI(Db*wr*(grad(b) + grad(b.Other())) * n * (tb - tb.Other()), skeleton=True)
            a += SymbolicBFI(Db*wr*(grad(tb) + grad(tb.Other())) * n * (b - b.Other()), skeleton=True)
            a += SymbolicBFI(-Db*2*wr*eta / h * (b - b.Other()) * (tb - tb.Other()), skeleton=True)

            a += SymbolicBFI(Db*b2*grad(r)*grad(tb))
            a += SymbolicBFI(-Db*wb*(grad(r) + grad(r.Other())) * n * (tb - tb.Other()), skeleton=True)
            a += SymbolicBFI(-Db*wb*(grad(tb) + grad(tb.Other())) * n * (r - r.Other()), skeleton=True)
            a += SymbolicBFI(Db*2*wb*eta / h * (r - r.Other()) * (tb - tb.Other()), skeleton=True)

            def abs(x):
                return IfPos(x, x, -x)

            # upwind scheme for the advection
            # missing boundary term??

            # equation for r
            a += SymbolicBFI(-r*velocityr*grad(tr))
            a += SymbolicBFI(velocityr*n*0.5*(r + r.Other())*(tr - tr.Other()), skeleton=True)
            a += SymbolicBFI(0.5*abs(velocityr*n) * (r - r.Other())*(tr - tr.Other()), skeleton=True)

            # equation for b
            a += SymbolicBFI(-b*velocityb*grad(tb))
            a += SymbolicBFI(velocityb*n*0.5*(b + b.Other())*(tb - tb.Other()), skeleton=True)
            a += SymbolicBFI(0.5*abs(velocityb*n) * (b - b.Other())*(tb - tb.Other()), skeleton=True)

            # mass matrix
            m = BilinearForm(fes)
            m += SymbolicBFI(r*tr)
            m += SymbolicBFI(b*tb)

            print('Assembling m...')
            m.Assemble()

            # Calculate constant equilibria
            domainSize = Integrate(CoefficientFunction(1),mesh,definedon=topMat)
            mr = Integrate(r2,mesh, definedon=topMat)
            mb = Integrate(b2,mesh, definedon=topMat)

            rbinfty = GridFunction(fes)
            rinfty = rbinfty.components[0]
            binfty = rbinfty.components[1]

            #rinfty = Integrate(r2,mesh, definedon=topMat) / domainSize
            #binfty = Integrate(b2,mesh, definedon=topMat) / domainSize 

            #### TODO: Add diffusion coeffs

            # Newton Solver to determine stationary solutions
            def AApply(uv,V,W,mesh):
                mmr = Integrate( exp((uv[0]-V)/Dr) / (1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db)), mesh, definedon=topMat )
                mmb = Integrate( exp((uv[1]-W)/Db) / (1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db)), mesh, definedon=topMat )
                #w = v * (1 - v) * (v - alpha)
                # print(dt * np.hstack((w, -w)))
                return (np.hstack((mmr, mmb)))

            def AssembleLinearization(uv,V,W,mesh):
                m = np.empty([2,2])
                m[0,0] = Integrate(exp((uv[0]-V)/Dr)*(1+exp((uv[1]-W)/Db))/((1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))*(1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))*Dr),mesh,definedon=topMat)
                m[0,1] = Integrate(-1/Db*exp((uv[0]-V)/Dr)*exp((uv[1]-W)/Db)/((1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))*(1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))),mesh,definedon=topMat)

                m[1,0] = Integrate(-1/Dr*exp((uv[0]-V)/Dr)*exp((uv[1]-W)/Db)/((1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))*(1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))),mesh,definedon=topMat)
                m[1,1] = Integrate(exp((uv[1]-W)/Db)*(1+exp((uv[0]-V)/Dr))/((1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))*(1+exp((uv[0]-V)/Dr)+exp((uv[1]-W)/Db))*Db),mesh,definedon=topMat)

                # print(dt * Alin.toarray())
                return m

            updnorm = 1e99
            #uinfty = 1
            #vinfty = 1
            uvinfty = np.hstack((0.0,0.0))
            # Newton solver
            while updnorm > 1e-9:
                rhs = AApply(uvinfty,Vr,Vb,mesh) - np.hstack((mr, mb))
                Alin = AssembleLinearization(uvinfty,Vr,Vb,mesh)
            #    urgh
                upd = np.linalg.solve(Alin, rhs)

                updnorm = np.linalg.norm(upd)

                uvinfty = uvinfty - 0.1*upd
                # input('')

            print('Newton converged with error' + '|w| = {:7.3e} '.format(updnorm),end='\n')
            rinfty.Set(exp((uvinfty[0]-gridr)/Dr) / (1+exp((uvinfty[0]-gridr)/Dr)+exp((uvinfty[1]-gridb)/Db)))
            binfty.Set(exp((uvinfty[1]-gridb)/Db) / (1+exp((uvinfty[0]-gridr)/Dr)+exp((uvinfty[1]-gridb)/Db)))

            rhs = s.vec.CreateVector()
            mstar = m.mat.CreateMatrix()


            # Draw(r2, mesh, 'r')
            # Draw(b2, mesh, 'b')
            # visualize both species at the same time, red in top rectangle, blue in bottom
            # translate density b2 of blue species to bottom rectangle
            both = r2 + Compose((x, y+1.3), b2, mesh)
            both2 = rinfty + Compose((x, y+1.3), binfty, mesh)
            Draw(both2, mesh, 'stationary')
            Draw(both, mesh, 'both')


            # times = [0.0]
            entropy = rinfty*ZLogZCF(r2/rinfty) + binfty*ZLogZCF(b2/binfty) + (1-rinfty-binfty)*ZLogZCF((1-r2-b2)/(1-rinfty-binfty)) + r2*gridr + b2*gridb
            ents = [Integrate(entropy, mesh, definedon=topMat)]
            # fig, ax = plt.subplots()
            # line, = ax.plot(times, ents)
            # plt.show(block=False)

            # input("Press any key...")
            # semi-implicit Euler
            t = 0.0
            with TaskManager():
                while tend < 0 or t < tend - tau / 2:
                    print("\nt = {:10.6e}".format(t))
                    t += tau

                    # print('Calculating convolution integrals...')
                    # with ConvolutionCache(convr), ConvolutionCache(convb):
                    #     gridr.Set(Vr+convr)
                    #     gridb.Set(Vb+convb)
                    print('Assembling a...')
                    a.Assemble()

                    rhs.data = m.mat * s.vec

                    mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
                    invmat = mstar.Inverse(fes.FreeDofs())
                    s.vec.data = invmat * rhs

                    Redraw(blocking=False)
                    # times.append(t)
                    ents.append(Integrate(entropy, mesh, definedon=topMat))
                    # line.set_xdata(times)
                    # line.set_ydata(ents)
                    # ax.relim()
                    # ax.autoscale_view()
                    # fig.canvas.draw()
                    # input()

            er = Integrate(pow(rinfty-r2, 2), mesh, definedon=topMat)
            eb = Integrate(pow(binfty-b2, 2), mesh, definedon=topMat)
            outfile = open('data/crossdiff/ents_'+str(order)+'_'+str(maxh)+'_'+str(eta)+'.csv', 'w')
            print('{}, {}, {}, {}, {}'.format(order, maxh, eta, er, eb))
            outfile.write('{}, {}\n'.format(er, eb))
            for item in ents:
                outfile.write('{}\n'.format(item))
            outfile.close()

