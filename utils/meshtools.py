"""
Some tools to make working with netgen meshes easier.
Stolen from gitlab:
https://gitlab.asc.tuwien.ac.at/jschoeberl/mri/tree/master/merge/merge
"""


def facedescriptor_list(mesh):
    """Return list of face descriptors of the mesh."""

    # Face descriptors are 0 based, but indexing by FaceDescriptor method is 1
    # based
    return [mesh.FaceDescriptor(fd_index) for fd_index in
            range(1, mesh.GetNFaceDescriptors()+1)]


def nr_bcs(mesh):
    """Return number of boundary conditions of the mesh."""

    # Boundary conditions are 1 based in face descriptors.
    return max((fd.bc for fd in facedescriptor_list(mesh)), default=0)


def bc_list(mesh):
    """Return list of boundary conditions names of the mesh."""

    # Boundary condition names are 0 based.
    return [mesh.GetBCName(bc_index) for bc_index in range(nr_bcs(mesh))]

def nr_materials(mesh):
    """Return number of materials belonging to the mesh.

    This assumes, that the number of domains and the number of materials
    coincide.  This is likely satisfied, because materials with the same name
    get listed twice for separate domains.  Could be substituted by
    GetNDomains() from netgen."""

    if mesh.dim == 2:
        # the method below doesn't work in 2D, because
        # SplineGeometry sets all domin, domout=0 => nr_materials=0
        # have to use bcs here
        return nr_bcs(mesh)
    else:
        return max((dom
                for fd in facedescriptor_list(mesh)
                for dom in (fd.domin, fd.domout)), default=0)


def materials_list(mesh):
    """Return list of materials of the mesh."""

    # Materials are 1 based.
    return [mesh.GetMaterial(mat) for mat in range(1, nr_materials(mesh)+1)]


def max_surfnr(mesh):
    """Return biggest surface number of occurring in mesh."""
    return max((fd.surfnr for fd in facedescriptor_list(mesh)), default=0)
