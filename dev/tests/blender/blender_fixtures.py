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
face_uvs = addon.src.utils.mesh.face_uvs

# the machine is often busy with a bench run
UNWRAP_SECONDS = 600
OUTPUT_SUFFIX = "_unwrapped"

needs_engine = pytest.mark.skipif(
    addon.src.utils.engine_binaries.get_dev_build_path("optcuts") is None,
    reason="no local optcuts binary, and downloading one would need the network",
)
needs_tools = pytest.mark.skipif(
    addon.src.utils.engine_binaries.get_dev_build_path("uvgami-tools") is None,
    reason="no local uvgami-tools binary, and no release exists to download",
)
needs_xatlas = pytest.mark.skipif(
    addon.src.utils.engine_binaries.get_dev_build_path("xatlas") is None,
    reason="no local xatlas binary, and downloading one would need the network",
)


def _partuv_installed():
    partuv = addon.src.engines.partuv
    if not partuv.ENGINE.is_installed():
        return False
    repo = partuv.find_partuv_dev_repo()
    venv = partuv.paths.get_partuv_venv_path() if repo is None else repo / ".venv"
    return any(
        next(venv.glob(pattern), None) is not None
        for pattern in ("*/site-packages/partuv-*", "*/*/site-packages/partuv-*")
    )


needs_partuv = pytest.mark.skipif(
    not _partuv_installed(), reason="partuv is not installed for this blender"
)


PROXY_FACES_MINIMUM = addon.src.ui.props.UVGAMI_PG_properties.bl_rna.properties[
    "proxy_faces"
].hard_min
# (islands, seams) on the balanced unwrap, recorded 2026-08-22 at optcuts 1.20,
# the hard surface rows 2026-10-24 on the placed seams route
EXPECTED = {
    ("cylinder", "ORGANIC"): (1, 30),
    ("cylinder", "HARD"): (3, 97),
    ("cube-bevel2", "ORGANIC"): (1, 12),
    ("cube-bevel2", "HARD"): (6, 37),
}


def uv_islands(mesh):
    faces = addon.src.utils.mesh.face_vertices(mesh)
    uvs = addon.src.utils.mesh.face_uvs(mesh)
    edges = addon.src.no_bpy.faces_by_edge(faces)
    return addon.src.no_bpy.islands_from_uvs(faces, uvs, edges)


def island_count(obj):
    return len(uv_islands(obj.data))


def seam_count(obj):
    return sum(1 for edge in obj.data.edges if edge.use_seam)


def uvs_inside_unit_square(mesh):
    uvs = addon.src.utils.mesh.loop_uvs(mesh)
    return bool(((uvs >= 0) & (uvs <= 1)).all())


def has_uvs(obj, faces=None):
    mesh = obj.data
    if mesh.uv_layers.active is None:
        return False
    # a map crushed to points has corners off zero but no area
    uvs = addon.src.utils.mesh.face_uvs(mesh)
    if faces is not None:
        uvs = [uvs[face.index] for face in faces]
    return not addon.src.no_bpy.uvs_collapsed_to_points(uvs)


def add_primitive(name, add, **kwargs):
    add(**kwargs)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_cube(name, **kwargs):
    return add_primitive(name, bpy.ops.mesh.primitive_cube_add, **kwargs)


def add_sphere(name):
    return add_primitive(
        name, bpy.ops.mesh.primitive_uv_sphere_add, segments=32, ring_count=16
    )


def add_dense_sphere(name, location=(0, 0, 0)):
    return add_primitive(
        name,
        bpy.ops.mesh.primitive_uv_sphere_add,
        segments=64,
        ring_count=32,
        location=location,
    )


def select(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def join(objects):
    select(objects)
    bpy.ops.object.join()
    return bpy.context.active_object


def two_loose_cubes():
    return join([add_cube("a"), add_cube("b", location=(5, 0, 0))])


def is_second_cube(mesh, face):
    # the second cube of two_loose_cubes sits past x = 2
    return mesh.vertices[face.vertices[0]].co.x > 2


def side_uvs(obj, second):
    mesh = obj.data
    layer = mesh.uv_layers.active.data
    return {
        tuple(round(coordinate, 5) for coordinate in layer[loop].uv)
        for face in mesh.polygons
        if is_second_cube(mesh, face) == second
        for loop in face.loop_indices
    }


@pytest.fixture(autouse=True)
def empty_scene():
    bpy.ops.wm.read_homefile(use_empty=True)


@pytest.fixture
def load_obj():
    def load(stem):
        path = FIXTURES / f"{stem}.obj"
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
        select(imported)
        return imported

    return load


# the progress bar needs a gpu context, a popup crashes a background blender
SESSION_PREFERENCES = {
    "autosave": False,
    "show_progress_bar": False,
    "show_popup": False,
}


@pytest.fixture
def session():
    preferences = get_preferences()
    saved = {name: getattr(preferences, name) for name in SESSION_PREFERENCES}
    for name, value in SESSION_PREFERENCES.items():
        setattr(preferences, name, value)
    try:
        with timer_pump.pump_timers() as pump:

            def run(operator, until=None, **kwargs):
                operator(**kwargs)
                done = until if until is not None else lambda: not manager.is_active
                pump.run_until(done, UNWRAP_SECONDS)
                return pump

            yield run
    finally:
        for name, value in saved.items():
            setattr(preferences, name, value)


@pytest.fixture
def unwrap(session):
    def run(engine="OPTCUTS", until=None):
        bpy.context.scene.uvgami.engine = engine
        return session(bpy.ops.uvgami.start, until)

    return run


@pytest.fixture
def outputs():
    def read():
        return {
            obj.name: obj
            for obj in bpy.data.objects
            if obj.name.endswith(OUTPUT_SUFFIX)
        }

    return read


@pytest.fixture
def invalid_objects():
    def read():
        collection = bpy.data.collections.get(
            addon.src.utils.mesh.NOT_UNWRAPPED_COLLECTION
        )
        if collection is None:
            return {}
        return {obj.name: obj for obj in collection.objects}

    return read


@pytest.fixture
def make_mesh():
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
def seam_edges():
    def read(obj):
        return {frozenset(edge.vertices) for edge in obj.data.edges if edge.use_seam}

    return read
