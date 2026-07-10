import importlib.util
import math
from pathlib import Path

# loaded from file so it doesn't need bench/ on sys.path
spec = importlib.util.spec_from_file_location(
    "bench_metrics", Path(__file__).parents[1] / "bench" / "metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def write_obj(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


# unit square, uv identical to xy: one chart, no seam, isometric mapping
ISOMETRIC = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""

# same geometry but the two triangles use disjoint vt indices across the shared
# 3D edge (verts 1-3), splitting the UV into two charts with a seam there
SEAM = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/4 3/5 4/6
"""

# uv stretched 2x along v only: areas stay proportional but angles distort
ANISOTROPIC = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 2
vt 0 2
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


def test_isometric_is_perfect(tmp_path):
    m = metrics.compute_metrics(write_obj(tmp_path, "iso.obj", ISOMETRIC))
    assert m["chart_count"] == 1
    assert m["seam_length"] == 0.0
    assert m["degenerate_tris"] == 0
    assert math.isclose(m["area_distortion"], 1.0, rel_tol=1e-9)
    assert math.isclose(m["angle_distortion"], 1.0, rel_tol=1e-9)
    assert math.isclose(m["uv_utilization"], 1.0, rel_tol=1e-9)


def test_seam_splits_charts(tmp_path):
    m = metrics.compute_metrics(write_obj(tmp_path, "seam.obj", SEAM))
    assert m["chart_count"] == 2
    # shared 3D edge has length sqrt(2), total 3D area is 1 so normalizer is 1
    assert math.isclose(m["seam_length"], math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(m["area_distortion"], 1.0, rel_tol=1e-9)


def test_anisotropic_stretch(tmp_path):
    m = metrics.compute_metrics(write_obj(tmp_path, "aniso.obj", ANISOTROPIC))
    # relative areas preserved, so area distortion stays 1
    assert math.isclose(m["area_distortion"], 1.0, rel_tol=1e-9)
    # L2 stretch of a 2x anisotropic map after global area-normalization
    assert math.isclose(m["angle_distortion"], math.sqrt(1.25), rel_tol=1e-9)


def test_degenerate_counted(tmp_path):
    body = ISOMETRIC + "f 1/1 2/2 2/2\n"  # zero-area triangle
    m = metrics.compute_metrics(write_obj(tmp_path, "deg.obj", body))
    assert m["degenerate_tris"] == 1
    assert math.isclose(m["area_distortion"], 1.0, rel_tol=1e-9)
