from netgen.geom2d import unit_square, SplineGeometry

# geometries for crossdiffusion
# return the number of domains used

def square(geo):
    geo.AddRectangle((0, 0), (1, 1))

    return 1

def window(geo):
    geo.AddRectangle((0, 0), (2, 1), leftdomain=1)
    geo.AddRectangle((0.2, 0.2), (0.9, 0.8), leftdomain=0, rightdomain=1)
    geo.AddRectangle((1.1, 0.2), (1.8, 0.8), leftdomain=0, rightdomain=1)

    return 1

def patchClamp(geo, maxh_corridor=0.05):
    points = [
        # left half
        (0.9, 0.6), (0.9, 1), (0, 1), (0, 0), (0.9, 0), (0.9, 0.4),
        # right half
        (1.1, 0.4), (1.1, 0), (2, 0), (2, 1), (1.1, 1), (1.1, 0.6)]

    pids = [geo.AppendPoint(*p) for p in points]

    # left half
    for p1, p2 in zip(pids[:5], pids[1:6]):
        geo.Append(['line', p1, p2])
    # right half
    for p1, p2 in zip(pids[6:-1], pids[7:]):
        geo.Append(['line', p1, p2])

    # center corridor
    geo.Append(['line', pids[0], pids[5]], leftdomain=2, rightdomain=1)
    geo.Append(['line', pids[5], pids[6]], leftdomain=2)
    geo.Append(['line', pids[6], pids[11]], leftdomain=2, rightdomain=1)
    geo.Append(['line', pids[11], pids[0]], leftdomain=2)

    # finer mesh for center corridor
    geo.SetDomainMaxH(2, 0.05)

    # circles
    # left half
    geo.AddCircle((0.6, 0.7), 0.07, leftdomain=0, rightdomain=1)
    geo.AddCircle((0.4, 0.4), 0.07, leftdomain=0, rightdomain=1)
    geo.AddCircle((0.5, 0.2), 0.07, leftdomain=0, rightdomain=1)

    # right half
    geo.AddCircle((1.4, 0.8), 0.07, leftdomain=0, rightdomain=1)
    geo.AddCircle((1.8, 0.6), 0.07, leftdomain=0, rightdomain=1)
    geo.AddCircle((1.6, 0.2), 0.07, leftdomain=0, rightdomain=1)

    return 2
