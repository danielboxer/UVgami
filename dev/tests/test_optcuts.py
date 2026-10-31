import collections
import math
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from uvgami_cli import optcuts
from uvgami_cli._pytest_plugin import FIXTURES, ONE_TRIANGLE_OBJ
from uvgami_cli.common import UnwrapError, engine_binaries, find_engine


class FakeProcess:
    def __init__(self, argv, returncode=0, output_text=ONE_TRIANGLE_OBJ):
        self.argv = argv
        self.returncode = returncode
        self.output_text = output_text
        self.stdout = iter(["progress: 0 0 1\n"])

    def wait(self):
        input_path = Path(self.argv[self.argv.index("-i") + 1])
        # captured here because the workdir is deleted after run() returns
        self.input_folder_names = {path.name for path in input_path.parent.iterdir()}
        if self.returncode == 0:
            output_dir = Path(self.argv[self.argv.index("-o") + 1])
            (output_dir / f"{input_path.stem}.obj").write_text(self.output_text)
        return self.returncode


def test_build_args_quality_and_weight_mapping(fake_engine):
    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "less-stretch", 5, False
    )
    assert args[args.index("-u") + 1] == "4.05"
    assert args[args.index("-s") + 1] == "200"
    assert "-g" in args

    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "fewer-seams", 1, True
    )
    assert args[args.index("-u") + 1] == "5.0"
    assert args[args.index("-s") + 1] == "25"
    assert "-g" not in args


def test_output_dir_arg_has_trailing_separator(fake_engine):
    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "balanced", 3, False
    )
    output_argument = args[args.index("-o") + 1]
    assert output_argument.endswith(("/", "\\"))


def test_run_success(triangle, tmp_path, fake_engine, popen_recorder):
    popen_recorder(FakeProcess)
    output = tmp_path / "result.obj"
    optcuts.run(triangle, output, "balanced", False, None, 3, fake_engine)
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_copies_weights_sidecar(triangle, tmp_path, fake_engine, popen_recorder):
    calls = popen_recorder(FakeProcess)
    weights = tmp_path / "weights.txt"
    weights.write_text("0,1.0")
    optcuts.run(
        triangle, tmp_path / "out.obj", "balanced", False, weights, 3, fake_engine
    )
    argv = calls[0].argv
    assert Path(argv[argv.index("-i") + 1]).name == "triangle.obj"
    assert "triangle_weights" in calls[0].input_folder_names


def test_run_copies_sidecars_next_to_the_input(
    triangle, tmp_path, fake_engine, popen_recorder
):
    calls = popen_recorder(FakeProcess)
    (tmp_path / "triangle_fixed").write_text("0,1")

    optcuts.run(triangle, tmp_path / "out.obj", "balanced", False, None, 3, fake_engine)
    assert "triangle_fixed" in calls[0].input_folder_names


def test_run_engine_failure(triangle, tmp_path, fake_engine, popen_recorder):
    popen_recorder(FakeProcess, returncode=7)
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle, tmp_path / "out.obj", "balanced", False, None, 3, fake_engine
        )
    assert error.value.exit_code == 4


def test_run_timeout_kills_the_engine(triangle, tmp_path, fake_engine, monkeypatch):
    class HangingProcess:
        def __init__(self, argv, **kwargs):
            self.returncode = 1
            self.killed = threading.Event()
            self.stdout = self._lines()

        def _lines(self):
            assert self.killed.wait(10), "the timer never killed the engine"
            yield from ()

        def kill(self):
            self.killed.set()

        def wait(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", HangingProcess)
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle,
            tmp_path / "out.obj",
            "balanced",
            False,
            None,
            3,
            fake_engine,
            0.05,
        )
    assert error.value.exit_code == 4
    assert "timed out" in str(error.value)


def test_run_output_missing_uvs(triangle, tmp_path, fake_engine, popen_recorder):
    popen_recorder(FakeProcess, output_text="v 0 0 0\nf 1 1 1\n")
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle, tmp_path / "out.obj", "balanced", False, None, 3, fake_engine
        )
    assert error.value.exit_code == 5


def test_find_engine_explicit_path_missing(tmp_path):
    with pytest.raises(UnwrapError) as error:
        find_engine("optcuts", "OptCuts", tmp_path / "nope.exe")
    assert error.value.exit_code == 3


def test_find_engine_reads_the_engine_dir_override(tmp_path, monkeypatch):
    binary = tmp_path / engine_binaries.get_engine_binary_name("optcuts")
    binary.write_text("fake")
    monkeypatch.setenv(engine_binaries.ENGINE_DIR_ENV_VAR, str(tmp_path))
    assert find_engine("optcuts", "OptCuts", None) == binary


def run_engine(binary, obj, output_dir):
    return subprocess.run(
        [str(binary), "-i", str(obj), "-o", str(output_dir) + os.sep],
        timeout=300,
        capture_output=True,
    )


def make_dirs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


def write_obj_with_planar_uvs(path, verts, faces):
    with path.open("w") as obj:
        for x, y, z in verts:
            obj.write(f"v {x!r} {y!r} {z!r}\n")
        for x, y, _ in verts:
            obj.write(f"vt {x!r} {y!r}\n")
        for face in faces:
            corners = " ".join(f"{corner + 1}/{corner + 1}" for corner in face)
            obj.write(f"f {corners}\n")


BumpPatch = collections.namedtuple("BumpPatch", "verts faces border")


def write_bump_patch(path, size=12, height=0.3, width=0.05):
    verts = []
    border = []
    for row in range(size + 1):
        for column in range(size + 1):
            x, y = column / size, row / size
            z = height * math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / width)
            verts.append((x, y, z))
            if column in (0, size) or row in (0, size):
                border.append(row * (size + 1) + column)
    faces = []
    for row in range(size):
        for column in range(size):
            corner = row * (size + 1) + column
            faces.append((corner, corner + 1, corner + size + 2))
            faces.append((corner, corner + size + 2, corner + size + 1))
    write_obj_with_planar_uvs(path, verts, faces)
    (path.parent / f"{path.stem}_fixed").write_text(",".join(map(str, border)))
    return BumpPatch(verts, faces, border)


def read_output(path, faces):
    # each input vt maps to its output vt through the face order the engine keeps
    uvs = []
    output_faces = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            uvs.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "f":
            output_faces.append(
                tuple(int(corner.split("/")[1]) - 1 for corner in parts[1:])
            )
    assert len(output_faces) == len(faces)
    output_uv_of = {}
    for input_face, output_face in zip(faces, output_faces):
        for input_uv, output_uv in zip(input_face, output_face):
            output_uv_of[input_uv] = output_uv
    return uvs, output_uv_of


def unscale_by_pins(verts, uvs, output_uv_of, border):
    # the output is uniformly scaled into the unit box, two pins undo it
    pin_pairs = [((verts[t][0], verts[t][1]), uvs[output_uv_of[t]]) for t in border]
    lowest = min(pin_pairs, key=lambda pair: pair[0][0])
    highest = max(pin_pairs, key=lambda pair: pair[0][0])
    scale = (highest[0][0] - lowest[0][0]) / (highest[1][0] - lowest[1][0])
    shift_u = lowest[0][0] - scale * lowest[1][0]
    shift_v = lowest[0][1] - scale * lowest[1][1]
    return lambda t: (
        scale * uvs[output_uv_of[t]][0] + shift_u,
        scale * uvs[output_uv_of[t]][1] + shift_v,
    )


def largest_offset(placed, verts, vertices):
    # the input uv of a vertex is its grid position
    return max(
        max(abs(placed(t)[0] - verts[t][0]), abs(placed(t)[1] - verts[t][1]))
        for t in vertices
    )


@pytest.mark.smoke
def test_optcuts_smoke(cube, tmp_path, engine_binary):
    engine_binary("optcuts")
    output = tmp_path / "cube_uv.obj"
    optcuts.run(cube, output, "fewer-seams", False, None, 3, None)
    text = output.read_text()
    assert "vt " in text
    assert "f " in text


@pytest.mark.smoke
def test_optcuts_shared_process_matches_solo(tmp_path, engine_binary):
    binary = engine_binary("optcuts")
    input_dir = tmp_path / "input"
    solo_dir = tmp_path / "solo"
    shared_dir = tmp_path / "shared"
    for directory in (input_dir, solo_dir, shared_dir):
        directory.mkdir()
    cube = input_dir / "cube.obj"
    shutil.copyfile(FIXTURES / "cube.obj", cube)
    patch = input_dir / "patch.obj"
    write_bump_patch(patch)
    # the pinned patch first, its pins must not leak into the cube
    for obj in (patch, cube):
        assert run_engine(binary, obj, solo_dir).returncode == 0
    result = subprocess.run(
        [str(binary), "-o", str(shared_dir) + os.sep],
        input=f"unwrap {patch}\nunwrap {cube}\nunwrap {input_dir / 'none.obj'}\n",
        text=True,
        timeout=300,
        capture_output=True,
    )
    assert result.returncode == 0
    markers = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("start:", "done:", "failed:"))
    ]
    assert markers == [
        "start: patch",
        "done: patch",
        "start: cube",
        "done: cube",
        "start: none",
        "failed: none 103",
    ]
    for obj in (patch, cube):
        assert (shared_dir / obj.name).read_bytes() == (
            solo_dir / obj.name
        ).read_bytes()


@pytest.mark.smoke
def test_optcuts_pinned_border(tmp_path, engine_binary):
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "patch.obj"
    verts, faces, border = write_bump_patch(obj)

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 0

    uvs, output_uv_of = read_output(output_dir / "patch.obj", faces)
    placed = unscale_by_pins(verts, uvs, output_uv_of, border)
    assert largest_offset(placed, verts, border) < 1e-6


@pytest.mark.smoke
def test_optcuts_nocut_relaxes_without_cutting(tmp_path, engine_binary):
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "patch.obj"
    verts, faces, border = write_bump_patch(obj, size=20, height=1.5, width=0.01)
    sidecar = input_dir / "patch_fixed"
    sidecar.write_text(sidecar.read_text() + "\nnocut")

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 0

    uvs, output_uv_of = read_output(output_dir / "patch.obj", faces)
    # a cut duplicates the uv verts along it
    assert len(uvs) == len(verts)
    placed = unscale_by_pins(verts, uvs, output_uv_of, border)
    assert largest_offset(placed, verts, border) < 1e-6
    # the flat grid map is distorted over the bump, relaxing must move it
    interior = set(range(len(verts))) - set(border)
    assert largest_offset(placed, verts, interior) > 1e-3


@pytest.mark.smoke
def test_optcuts_nocut_without_pins(tmp_path, engine_binary):
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "patch.obj"
    verts, faces, _ = write_bump_patch(obj, size=20, height=1.5, width=0.01)
    (input_dir / "patch_fixed").write_text("\nnocut")

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 0

    uvs, output_uv_of = read_output(output_dir / "patch.obj", faces)
    # a cut duplicates the uv verts along it
    assert len(uvs) == len(verts)
    # input is the unit grid and the output is normalized into the unit box
    assert (
        largest_offset(lambda t: uvs[output_uv_of[t]], verts, range(len(verts))) > 1e-3
    )


@pytest.mark.smoke
def test_optcuts_nocut_keeps_hole_chart(tmp_path, engine_binary):
    size = 8
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "ring.obj"

    verts = []
    for row in range(size + 1):
        for column in range(size + 1):
            x, y = column / size, row / size
            z = 0.3 * math.exp(-((x - 0.2) ** 2 + (y - 0.2) ** 2) / 0.05)
            verts.append((x, y, z))
    hole = {size // 2 - 1, size // 2}
    faces = []
    for row in range(size):
        for column in range(size):
            if column in hole and row in hole:
                continue
            corner = row * (size + 1) + column
            faces.append((corner, corner + 1, corner + size + 2))
            faces.append((corner, corner + size + 2, corner + size + 1))
    # the vert inside the hole is unreferenced, drop it so vt counts compare
    used = sorted({v for face in faces for v in face})
    local = {v: index for index, v in enumerate(used)}
    verts = [verts[v] for v in used]
    faces = [tuple(local[v] for v in face) for face in faces]
    write_obj_with_planar_uvs(obj, verts, faces)
    (input_dir / "ring_fixed").write_text("\nnocut")

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 0

    uvs, _ = read_output(output_dir / "ring.obj", faces)
    # cutting the hole open to a disk would duplicate uv verts
    assert len(uvs) == len(verts)


@pytest.mark.smoke
def test_optcuts_keeps_a_flat_fan_with_straight_ends(tmp_path, engine_binary):
    # float noise puts the pieces of each straight end a hair off one line
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "fan.obj"
    rows, columns, span = 3, 12, math.radians(315.5)
    verts = []
    for column in range(columns + 1):
        angle = span * column / columns
        for row in range(rows + 1):
            radius = 1.0 + 0.1 * row
            verts.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    faces = []
    for column in range(columns):
        for row in range(rows):
            corner = column * (rows + 1) + row
            faces.append((corner, corner + 1, corner + rows + 2))
            faces.append((corner, corner + rows + 2, corner + rows + 1))
    write_obj_with_planar_uvs(obj, verts, faces)

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 0

    # the input map is already isometric, so a kept map comes back only scaled
    uvs, output_uv_of = read_output(output_dir / "fan.obj", faces)
    inputs = [math.dist(verts[0][:2], verts[t][:2]) for t in range(len(verts))]
    outputs = [
        math.dist(uvs[output_uv_of[0]], uvs[output_uv_of[t]]) for t in range(len(verts))
    ]
    scale = max(outputs) / max(inputs)
    largest_error = max(
        abs(output_distance - scale * input_distance)
        for output_distance, input_distance in zip(outputs, inputs)
    )
    assert largest_error < 1e-5 * max(outputs)


def drag_center_uv(obj, verts):
    # dragging one interior uv far away inverts its faces
    lines = obj.read_text().splitlines()
    lines[len(verts) + len(verts) // 2] = "vt -5 -5"
    obj.write_text("\n".join(lines) + "\n")


@pytest.mark.smoke
def test_optcuts_nocut_rejects_broken_map(tmp_path, engine_binary):
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "patch.obj"
    verts, _, _ = write_bump_patch(obj)
    (input_dir / "patch_fixed").write_text("\nnocut")
    drag_center_uv(obj, verts)

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 114


@pytest.mark.smoke
def test_optcuts_pinned_rejects_broken_map(tmp_path, engine_binary):
    input_dir, output_dir = make_dirs(tmp_path)
    obj = input_dir / "patch.obj"
    verts, _, _ = write_bump_patch(obj)
    drag_center_uv(obj, verts)

    assert run_engine(engine_binary("optcuts"), obj, output_dir).returncode == 110
