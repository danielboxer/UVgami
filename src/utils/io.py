import bpy


def split_shared_uvs(obj):
    """Nudge apart different vertices that landed on the same UV point.

    The obj exporter merges identical UVs into one vt, and optcuts rebuilds a
    UV-carrying mesh with one vertex per vt, so a UV shared by two 3D vertices
    welds them and degenerates the rebuilt mesh (the engine exits -1). Nudge
    per (vertex, uv) group so a vertex's corners stay welded to each other.
    Compare at export precision (6 decimals) and nudge past it."""
    layer = obj.data.uv_layers.active.data
    groups = {}  # (vertex, rounded uv) -> loop indices
    for poly in obj.data.polygons:
        for li in poly.loop_indices:
            uv = layer[li].uv
            vert = obj.data.loops[li].vertex_index
            key = (vert, (round(uv.x, 6), round(uv.y, 6)))
            groups.setdefault(key, []).append(li)

    taken = {}  # rounded uv -> vertex that owns it
    fixed = 0
    for (vert, uv_key), loops in groups.items():
        owner = taken.setdefault(uv_key, vert)
        if owner == vert:
            continue
        step = 1
        while True:
            nudged = (round(uv_key[0] + step * 1e-4, 6), uv_key[1])
            if taken.setdefault(nudged, vert) == vert:
                break
            step += 1
        for li in loops:
            layer[li].uv.x = nudged[0]
        fixed += 1
    return fixed


def export_obj(obj, path, export_uv):
    if export_uv and obj.data.uv_layers.active:
        split_shared_uvs(obj)
    obj.select_set(True)

    bpy.ops.wm.obj_export(
        "EXEC_DEFAULT",
        filepath=str(path),
        export_selected_objects=True,
        export_normals=False,
        export_uv=export_uv,
        export_materials=False,
        apply_modifiers=False,
        forward_axis="Y",
        up_axis="Z",
    )

    obj.select_set(False)


def import_obj(path, name=""):
    before = set(bpy.context.scene.objects)
    previous_select = bpy.context.selected_objects

    bpy.ops.wm.obj_import(
        "EXEC_DEFAULT", filepath=str(path), forward_axis="Y", up_axis="Z"
    )
    after = set(bpy.context.scene.objects)

    # get imported object
    imported_obj = after.difference(before).pop()
    imported_obj.select_set(False)

    for obj in previous_select:
        obj.select_set(True)

    if name != "":
        imported_obj.name = name

    return imported_obj


def print_stdin(process, msg):
    if process.poll() is not None:
        return False
    try:
        print(msg, file=process.stdin, flush=True)
    except OSError:
        return False
    return True
