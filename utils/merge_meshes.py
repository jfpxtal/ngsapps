"""
Merge two meshes to one mesh.
Stolen from gitlab:
https://gitlab.asc.tuwien.ac.at/jschoeberl/mri/tree/master/merge/merge
"""


from netgen.meshing import Mesh, FaceDescriptor, Element1D, Element2D, MeshPoint
from netgen.csg import Pnt

from .meshtools import max_surfnr, nr_bcs, nr_materials


def merge_meshes(mesh1, mesh2, offset1=(0, 0, 0), offset2=(0, 0, 0),
                 transfer_mats1=True, transfer_mats2=True):
    """Return a merged netgen mesh consisting of mesh1 and mesh2.

    Be aware of using the correct mesh objects.  The input and output mesh
    types are netgen meshes and NOT ngsolve meshes.

    Mesh1 needs to contain an enclosing solid, such as a sphere or cube.
    Materials and boundary conditions of mesh1 keep their number and those of
    mesh2 are appended.
    """

    if mesh1.dim != mesh2.dim:
        raise ValueError("meshes should have the same dimension")

    res_mesh = Mesh(mesh1.dim)

    transfer_facedescriptors(res_mesh, mesh1)
    transfer_facedescriptors(res_mesh, mesh2)

    if transfer_mats1:
        transfer_materials(res_mesh, mesh1)
    if transfer_mats2:
        transfer_materials(res_mesh, mesh2,
                        mat_offset=nr_materials(mesh1) if transfer_mats1 else 0)

    transfer_elements(res_mesh, mesh1, loc_offset=offset1)
    transfer_elements(res_mesh, mesh2,
                      fd_index_offset=mesh1.GetNFaceDescriptors(),
                      loc_offset=offset2)

    return res_mesh


def transfer_facedescriptors(mesh_to, mesh_from):
    """Add all face descriptors from mesh_from to mesh_to."""

    bc_offset = nr_bcs(mesh_to)
    domain_offset = nr_materials(mesh_to)
    surfnr_offset = max_surfnr(mesh_to)

    # Face descriptors are accessed 1 based.
    for fd_index in range(1, mesh_from.GetNFaceDescriptors()+1):
        from_fd = mesh_from.FaceDescriptor(fd_index)

        to_domin = from_fd.domin + domain_offset
        to_domout = from_fd.domout + domain_offset
        to_surfnr = from_fd.surfnr + surfnr_offset
        to_bc = from_fd.bc + bc_offset
        to_fd = FaceDescriptor(domin=to_domin, domout=to_domout,
                               surfnr=to_surfnr, bc=to_bc)

        mesh_to.Add(to_fd)

        # Boundary condition names are 0 based for mesh and 1 based for face
        # descriptors
        mesh_to.SetBCName(to_fd.bc-1, from_fd.bcname)


def transfer_materials(mesh_to, mesh_from, mat_offset=0):
    """Add all materials from mesh_from to mesh_to."""

    # Materials are 1 based.
    for domain_nr in range(1, nr_materials(mesh_from)+1):
        mesh_to.SetMaterial(domain_nr+mat_offset,
                            mesh_from.GetMaterial(domain_nr))


def transfer_elements(mesh_to, mesh_from, fd_index_offset=0, loc_offset=(0, 0, 0)):
    """Add all elements with vertices from mesh_from to mesh_to."""
    vertex_map = dict()

    # for elem in mesh_from.Elements1D():
    #     pids = []
    #     for vertex in elem.vertices:
    #         if vertex not in vertex_map:
    #             point = mesh_from[vertex]
    #             vertex_map[vertex] = mesh_to.Add(MeshPoint(Pnt(point[0]+loc_offset[0],
    #                                                            point[1]+loc_offset[1],
    #                                                            point[2]+loc_offset[2])))
    #         pids.append(vertex_map[vertex])
    #     # not sure how face descriptors relate to 1D Elements
    #     mesh_to.Add(Element1D(pids, index=elem.index + fd_index_offset))

    for elem in mesh_from.Elements2D():
        pids = []
        for vertex in elem.vertices:
            if vertex not in vertex_map:
                point = mesh_from[vertex]
                vertex_map[vertex] = mesh_to.Add(MeshPoint(Pnt(point[0]+loc_offset[0],
                                                               point[1]+loc_offset[1],
                                                               point[2]+loc_offset[2])))
            pids.append(vertex_map[vertex])
        mesh_to.Add(Element2D(elem.index + fd_index_offset, pids))

__all__ = [name for name, thing in locals().items() if callable(thing) and thing.__module__ == __name__]
