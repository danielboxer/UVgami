import numpy

from .seams import SURFACE_MODE_CODES

# face attributes, 0 on every one means the part uses the panel setting
SURFACE_MODE_ATTRIBUTE = "uvgami_surface_mode"
KEEP_ATTRIBUTE = "uvgami_keep"
# # 1 + the index into PRIORITY_ORDER
# PRIORITY_ATTRIBUTE = "uvgami_priority"
# UV_SCALE_ATTRIBUTE = "uvgami_uv_scale"

OVERRIDE_ATTRIBUTES = {
    SURFACE_MODE_ATTRIBUTE: "INT",
    KEEP_ATTRIBUTE: "INT",
    # PRIORITY_ATTRIBUTE: "INT",
    # UV_SCALE_ATTRIBUTE: "FLOAT",
}

# PRIORITY_ORDER = ("LESS_STRETCH", "BALANCED", "FEWER_SEAMS")


# object mode only, an edit mode mesh reads stale values
def face_values(mesh, name):
    # dtype = numpy.float32 if OVERRIDE_ATTRIBUTES[name] == "FLOAT" else numpy.int32
    values = numpy.zeros(len(mesh.polygons), dtype=numpy.int32)
    attribute = mesh.attributes.get(name)
    if attribute is None:
        return values
    if attribute.domain != "FACE":
        raise ValueError(f"{name} is a {attribute.domain} attribute, not a FACE one")
    attribute.data.foreach_get("value", values)
    return values


def surface_modes(mesh, unset_mode):
    modes = face_values(mesh, SURFACE_MODE_ATTRIBUTE)
    modes[modes == 0] = SURFACE_MODE_CODES[unset_mode]
    modes[face_values(mesh, KEEP_ATTRIBUTE) != 0] = SURFACE_MODE_CODES["ORGANIC"]
    return modes


# a separated piece is one loose part, and the panel writes whole parts
def piece_keeps_uvs(mesh):
    return bool(face_values(mesh, KEEP_ATTRIBUTE)[:1].any())


# def piece_priority_override(mesh):
#     code = int(face_values(mesh, PRIORITY_ATTRIBUTE)[:1].sum())
#     if code == 0:
#         return None
#     return PRIORITY_ORDER[code - 1]


def read_overrides(mesh):
    return {
        name: face_values(mesh, name).tolist()
        for name in OVERRIDE_ATTRIBUTES
        if name in mesh.attributes
    }


def write_overrides(mesh, overrides):
    for name, values in overrides.items():
        attribute = mesh.attributes.get(name) or mesh.attributes.new(
            name, OVERRIDE_ATTRIBUTES[name], "FACE"
        )
        attribute.data.foreach_set("value", values)
