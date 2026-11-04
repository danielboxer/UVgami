from collections import namedtuple

import bpy

from ..engines import ENGINES, active_engine
from ..no_bpy import EngineError
from ..proxy import needs_proxy, part_triangles, proxy_trigger_faces, triangle_count
from ..ui.props import describe_settings, unwrap_settings
from ..unwrap.handler import handle_start_error
from ..unwrap.logger import logger
from ..unwrap.manager import manager
from ..unwrap.session import MeshInput, SessionBuilder
from ..utils.mesh import deselect_all, is_shading_modifier
from ..utils.paths import get_preferences
from ..utils.ui import tag_redraw
from ..utils.uvgami_tools import tools_command

PreparedMeshes = namedtuple("PreparedMeshes", ["inputs", "reports", "input_sizes"])


class UVGAMI_OT_start(bpy.types.Operator):
    bl_idname = "uvgami.start"
    bl_label = "Unwrap"
    bl_description = "Start UV unwrap process"

    def execute(self, context):
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        try:
            if engine is None:
                ENGINES["OPTCUTS"].validate()
                raise EngineError("The selected engine is not installed")
            engine_ctx = engine.validate()
            segment_command = tools_command() if engine.splits_regions(props) else None
        except EngineError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if manager.is_active:
            self.report({"ERROR"}, "Finish or cancel the current unwrap first")
            return {"CANCELLED"}
        if self._autosave_blocks(get_preferences()):
            return {"CANCELLED"}

        start_objects = set(bpy.data.objects)
        temp_collection = None
        builder_registered = False

        try:
            info = logger.new_info()
            info.engine = engine.describe()
            info.settings = describe_settings(props, unwrap_settings(props))

            # leaving edit mode flushes the mesh the copies below take
            active = context.active_object
            if active is not None and active.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            prepared = prepare_meshes(context, engine)
            info.objects = prepared.input_sizes
            inputs = prepared.inputs
            if not inputs:
                reasons = prepared.reports
                reason = reasons[0][1] if reasons else "No object selected"
                self.report({"ERROR"}, reason)
                logger.discard_refused_run()
                return {"CANCELLED"}

            input_path = manager.prepare_input_dir()
            deselect_all()

            # unlinked collections don't show in the viewport between ticks
            temp_collection = bpy.data.collections.new("UVgami Temp")
            for entry in inputs:
                for collection in entry.copy.users_collection:
                    collection.objects.unlink(entry.copy)
                temp_collection.objects.link(entry.copy)

            builder = SessionBuilder(
                engine=engine,
                segment_command=segment_command,
                input_path=input_path,
                inputs=inputs,
                start_objects=start_objects,
                temp_collection=temp_collection,
            )
            bpy.app.timers.register(builder.tick)
            builder_registered = True

            manager.pieces_still_arriving += 1
            manager.start(engine, engine_ctx)
            tag_redraw()

            for level, text in prepared.reports:
                self.report({level}, text)

        except Exception as error:
            handle_start_error(error, start_objects)
            if not builder_registered and temp_collection is not None:
                bpy.data.collections.remove(temp_collection)

        return {"FINISHED"}

    def _autosave_blocks(self, prefs):
        if not prefs.autosave:
            return False

        if bpy.data.is_saved:
            bpy.ops.wm.save_mainfile()
            return False

        bpy.ops.wm.save_as_mainfile("INVOKE_DEFAULT")
        self.report(
            {"WARNING"},
            "Autosave is turned on. Save the file before starting UVgami",
        )
        return True


def prepare_meshes(context, engine):
    props = context.scene.uvgami

    inputs = []
    input_sizes = []
    skipped = set()
    seen_mesh_data = set()
    applied_modifiers = False
    unapplied_modifiers = []
    depsgraph = context.evaluated_depsgraph_get()
    for obj in context.selected_objects:
        if obj.type != "MESH":
            skipped.add("non mesh objects")
            continue
        if len(obj.data.polygons) == 0:
            skipped.add("objects with zero polygons")
            continue
        if props.transfer_uvs and obj.data in seen_mesh_data:
            skipped.add("duplicates sharing a mesh")
            continue
        seen_mesh_data.add(obj.data)

        copy_object = obj.copy()
        copy_object.data = obj.data.copy()
        copy_object.animation_data_clear()

        obj.users_collection[0].objects.link(copy_object)

        evaluated = obj.evaluated_get(depsgraph)
        triangles = triangle_count(evaluated)
        input_sizes.append((obj.name, triangles))
        parts = part_triangles(copy_object.data) if engine.uses_proxy(props) else None
        proxied = parts is not None and needs_proxy(
            copy_object, proxy_trigger_faces(props), parts
        )
        if proxied or props.transfer_uvs:
            copy_object.modifiers.clear()
        else:
            applied, failed = _apply_modifiers(context, copy_object)
            applied_modifiers |= applied
            unapplied_modifiers += [f"{obj.name} {failure}" for failure in failed]

        inputs.append(MeshInput(copy_object, obj, obj.name, parts if proxied else None))

    reports = []
    if skipped:
        reports.append(("WARNING", f"Input contains {', '.join(sorted(skipped))}"))
    if applied_modifiers:
        reports.append(("WARNING", "Modifiers are applied to the unwrapped copy"))
    if unapplied_modifiers:
        reports.append(
            ("WARNING", f"Modifiers not applied: {', '.join(unapplied_modifiers)}")
        )

    return PreparedMeshes(inputs, reports, input_sizes)


def _apply_modifiers(context, obj):
    context.view_layer.objects.active = obj
    applied = False
    failed = []
    # applying one frees its entry in obj.modifiers
    names = [
        modifier.name for modifier in obj.modifiers if not is_shading_modifier(modifier)
    ]
    for name in names:
        # blender can't apply modifiers hidden in the viewport
        if not obj.modifiers[name].show_viewport:
            continue
        try:
            bpy.ops.object.modifier_apply(modifier=name)
        except RuntimeError as error:
            failed.append(f"{name}: {str(error).strip()}")
            continue
        applied = True
    return applied, failed
