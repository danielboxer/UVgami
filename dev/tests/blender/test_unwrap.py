import bpy
import pytest
from blender_fixtures import manager, needs_engine, needs_partuv, needs_xatlas

pytestmark = [needs_engine, pytest.mark.smoke]

OTHER_ENGINES = [
    pytest.param("XATLAS", marks=needs_xatlas),
    pytest.param("PARTUV", marks=needs_partuv),
]


def test_unwrap_runs_to_completion(load_obj, unwrap, outputs):
    load_obj("cylinder")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert list(outputs()) == ["cylinder_unwrapped"]
    layer = outputs()["cylinder_unwrapped"].data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)

    # a finished piece holding its three pipes open runs a many-part model
    # out of file descriptors partway through
    assert manager.results
    for piece, _ in manager.results:
        if piece.process is None:
            continue
        assert piece.process.returncode is not None
        assert piece.process.stdout.closed
        assert piece.process.stderr.closed


def test_flipped_face_is_rewound_instead_of_refused(load_obj, unwrap, outputs):
    """One face wound backwards used to come back as the engine's 115
    inconsistent-orientation refusal, the export rewinds it now."""
    load_obj("flipped-face")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert len(outputs()) == 1
    layer = next(iter(outputs().values())).data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)


@pytest.mark.parametrize("engine", OTHER_ENGINES)
def test_other_engines_unwrap_the_cylinder(load_obj, unwrap, outputs, engine):
    load_obj("cylinder")
    if engine == "PARTUV":
        bpy.context.scene.uvgami.partuv.segmentation = "GEOMETRIC"

    unwrap(engine)

    assert manager.summary == ["UV unwrap complete!"]
    assert manager.error_messages == []
    layer = outputs()["cylinder_unwrapped"].data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)
