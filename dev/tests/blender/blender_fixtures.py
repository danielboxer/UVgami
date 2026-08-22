"""Fixtures for tests that run inside Blender, imported by the conftest of
every folder run.py launches."""

import importlib
import pathlib

import background_blender
import bpy
import pytest
import timer_pump

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
addon = importlib.import_module(background_blender.ADDON_MODULE)
manager = addon.src.manager.manager
logger = addon.src.logger.logger
get_preferences = addon.src.utils.paths.get_preferences

# the machine is often busy with a bench run, so this is nowhere near a timing
UNWRAP_SECONDS = 600
OUTPUT_SUFFIX = "_unwrapped"
INVALID_COLLECTION = "UVgami Not Unwrapped"

needs_engine = pytest.mark.skipif(
    addon.src.utils.paths.get_local_engine_path("optcuts") is None,
    reason="no local optcuts binary, and downloading one would need the network",
)
needs_xatlas = pytest.mark.skipif(
    addon.src.utils.paths.get_local_engine_path("xatlas") is None,
    reason="no local xatlas binary, and downloading one would need the network",
)


def _partuv_installed():
    """The engine offers its dev route whenever dev/uvgami_cli, .venv and uv are
    all present, which is true in CI, where the wheel itself is not."""
    engine = addon.src.engines.get_engine("PARTUV")
    ctx, error = engine.validate(get_preferences())
    if error is not None:
        return False
    venv = ctx.path / ".venv" if ctx.mode == "dev" else ctx.path
    return any(
        next(venv.glob(pattern), None) is not None
        for pattern in ("*/site-packages/partuv-*", "*/*/site-packages/partuv-*")
    )


needs_partuv = pytest.mark.skipif(
    not _partuv_installed(), reason="partuv is not installed for this blender"
)


def island_count(obj):
    """How many uv islands the addon's own grouping finds on a mesh."""
    mesh = obj.data
    faces = addon.src.utils.mesh.face_vertices(mesh)
    uvs = addon.src.utils.mesh.face_uvs(mesh)
    edges = addon.src.seams.face_edges(faces)
    return len(addon.src.seams.uv_island_groups(faces, uvs, edges))


def seam_count(obj):
    return sum(1 for edge in obj.data.edges if edge.use_seam)


@pytest.fixture(autouse=True)
def empty_scene():
    """A blank file per test. Scene properties come back at their defaults
    with it, so one test's margin can't reach the next."""
    bpy.ops.wm.read_homefile(use_empty=True)


@pytest.fixture
def load_obj():
    """Import an obj by its stem from dev/tests/fixtures and leave it the only
    selection."""

    def load(stem):
        path = FIXTURES / f"{stem}.obj"
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
        bpy.ops.object.select_all(action="DESELECT")
        for obj in imported:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported[0]
        return imported

    return load


@pytest.fixture
def session():
    """Call an operator that starts a queue, then pump until the session is
    over. until() stops the pump early, and the returned pump keeps working
    until the test ends."""
    prefs = get_preferences()
    prefs.autosave = False
    # the progress bar needs a gpu context and a popup crashes a background
    # blender
    prefs.show_progress_bar = False
    prefs.show_popup = False
    with timer_pump.pump_timers() as pump:

        def run(operator, until=None, **kwargs):
            operator(**kwargs)
            done = until if until is not None else lambda: not manager.is_active
            pump.run_until(done, UNWRAP_SECONDS)
            return pump

        yield run


@pytest.fixture
def unwrap(session):
    """Start the unwrap operator on the selection and run it to the end."""

    def run(engine="OPTCUTS", until=None):
        bpy.context.scene.uvgami.engine = engine
        return session(bpy.ops.uvgami.start, until)

    return run


@pytest.fixture
def outputs():
    """The objects an unwrap produced, by name."""

    def read():
        return {
            obj.name: obj
            for obj in bpy.data.objects
            if obj.name.endswith(OUTPUT_SUFFIX)
        }

    return read


@pytest.fixture
def invalid_objects():
    """What was moved to the not unwrapped collection, by name."""

    def read():
        collection = bpy.data.collections.get(INVALID_COLLECTION)
        if collection is None:
            return {}
        return {obj.name: obj for obj in collection.objects}

    return read


@pytest.fixture
def make_mesh():
    """Build an object from verts, faces and one uv per face corner, then
    make it the active selection the operators expect."""

    def build(name, verts, faces, uvs):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        layer = mesh.uv_layers.new()
        for face, corners in zip(mesh.polygons, uvs):
            for loop_index, uv in zip(face.loop_indices, corners):
                layer.data[loop_index].uv = uv
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        return obj

    return build


@pytest.fixture
def face_uvs():
    """Per face, its corner uvs as rounded tuples, so two islands can be
    compared for the exact equality a stack needs."""

    def read(obj, digits=6):
        layer = obj.data.uv_layers.active.data
        return [
            [
                tuple(round(v, digits) for v in layer[loop].uv)
                for loop in face.loop_indices
            ]
            for face in obj.data.polygons
        ]

    return read


@pytest.fixture
def seam_edges():
    """The vertex index pairs blender has marked as seams."""

    def read(obj):
        return {frozenset(edge.vertices) for edge in obj.data.edges if edge.use_seam}

    return read
