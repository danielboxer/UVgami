import bmesh
import mathutils

from .mesh import new_bmesh, set_bmesh


def set_origin(obj, point):
    mw = obj.matrix_world
    obj.data.transform(mathutils.Matrix.Translation(-(mw.inverted() @ point)))
    mw.translation += point - mw.translation


def applied_matrix(obj):
    location, _, scale = obj.matrix_basis.decompose()
    return (
        mathutils.Matrix.Translation(location)
        @ obj.matrix_basis.to_3x3().normalized().to_4x4()
        @ mathutils.Matrix.Diagonal(scale).to_4x4()
    )


# bisecting just past a vertex near the plane leaves needle faces
PLANE_SNAP = 0.0001


# kept_side is 1 to keep the positive half, -1 for the negative one
def cut_on_axis(obj, obj_center, axis, snap, kept_side):
    bm = new_bmesh(obj)
    index = "XYZ".index(axis)
    for vert in bm.verts:
        if abs(vert.co[index] - obj_center[index]) < snap:
            vert.co[index] = obj_center[index]
    direction = [0, 0, 0]
    direction[index] = kept_side
    # a snapped float32 vertex misses the plane by rounding and gets a duplicate at dist 0
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        plane_co=obj_center,
        plane_no=direction,
        dist=snap,
        clear_inner=True,
    )
    set_bmesh(bm, obj)
