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

    if bpy.app.version >= (3, 1, 0):
        # new obj exporter
        args = {
            "filepath": str(path),
            "export_selected_objects": True,
            "export_normals": False,
            "export_uv": export_uv,
            "export_materials": False,
            "apply_modifiers": False,
            "forward_axis": "Y",
            "up_axis": "Z",
        }

        if bpy.app.version < (3, 2, 0):
            # apply_modifiers doesn't exist in 3.1
            del args["apply_modifiers"]

        if bpy.app.version < (3, 3, 0):
            # axis enums were renamed
            args["forward_axis"] = "Y_FORWARD"
            args["up_axis"] = "Z_UP"

        bpy.ops.wm.obj_export("EXEC_DEFAULT", **args)

    else:
        # old
        bpy.ops.export_scene.obj(
            "EXEC_DEFAULT",
            filepath=str(path),
            use_selection=True,
            use_normals=False,
            use_uvs=export_uv,
            use_materials=False,
            use_blen_objects=False,
            use_mesh_modifiers=False,
            axis_forward="Y",
            axis_up="Z",
        )

    obj.select_set(False)


def import_obj(path, name=""):
    before = set(bpy.context.scene.objects)
    previous_select = bpy.context.selected_objects

    # blender 3.1 doesn't have the new importer
    if bpy.app.version >= (3, 2, 0):
        # new obj importer
        forward = "Y"
        up = "Z"
        if bpy.app.version < (3, 3, 0):
            # axis names renamed
            forward = "Y_FORWARD"
            up = "Z_UP"

        bpy.ops.wm.obj_import(
            "EXEC_DEFAULT", filepath=str(path), forward_axis=forward, up_axis=up
        )

    else:
        # old
        bpy.ops.import_scene.obj(
            "EXEC_DEFAULT",
            filepath=str(path),
            split_mode="OFF",
            axis_forward="Y",
            axis_up="Z",
        )
    # push to undo stack
    bpy.ops.ed.undo_push()

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
