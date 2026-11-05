import math

import numpy
from uvgami_cli._pytest_plugin import corners_and_totals

from no_bpy.edge_loops import (
    EdgeLoops,
    LoopRegion,
    drawn_loops,
    loop_side,
    untouching_loops,
)
from no_bpy.segmentation import edge_face_pairs

RINGS = 12
SEGMENTS = 24
RING_SPACING = 0.3


def ring_vertex(ring, segment):
    return ring * SEGMENTS + segment % SEGMENTS


def capped_cylinder(triangles, jitter=0.0, ring_spacing=RING_SPACING):
    angles = 2 * math.pi * numpy.arange(SEGMENTS) / SEGMENTS
    heights = numpy.repeat(numpy.arange(RINGS) * ring_spacing, SEGMENTS)
    positions = numpy.stack(
        [
            numpy.tile(numpy.cos(angles), RINGS),
            numpy.tile(numpy.sin(angles), RINGS),
            heights,
        ],
        axis=1,
    )
    positions += numpy.random.default_rng(0).normal(0, jitter, positions.shape)
    bottom, top = len(positions), len(positions) + 1
    positions = numpy.vstack(
        [positions, [(0, 0, -ring_spacing), (0, 0, RINGS * ring_spacing)]]
    )
    faces = []
    for ring in range(RINGS - 1):
        for segment in range(SEGMENTS):
            a, b = ring_vertex(ring, segment), ring_vertex(ring, segment + 1)
            c, d = ring_vertex(ring + 1, segment + 1), ring_vertex(ring + 1, segment)
            if not triangles:
                faces.append([a, b, c, d])
            elif (ring + segment) % 2:
                faces += [[a, b, c], [a, c, d]]
            else:
                faces += [[a, b, d], [b, c, d]]
    for segment in range(SEGMENTS):
        faces.append([ring_vertex(0, segment + 1), ring_vertex(0, segment), bottom])
        faces.append(
            [ring_vertex(RINGS - 1, segment), ring_vertex(RINGS - 1, segment + 1), top]
        )
    return positions, *corners_and_totals(faces)


def loop_region(mesh, in_region):
    return LoopRegion(EdgeLoops(*mesh), in_region, edge_face_pairs(*mesh[1:]))


def loop_through(mesh, in_region, tail, head):
    return EdgeLoops(*mesh).find_loop(loop_region(mesh, in_region), tail, head)


def test_a_quad_loop_runs_around_the_cylinder():
    mesh = capped_cylinder(triangles=False)
    everywhere = numpy.ones(len(mesh[2]), dtype=bool)

    loop = loop_through(mesh, everywhere, ring_vertex(6, 3), ring_vertex(6, 4))

    assert sorted(loop) == [ring_vertex(6, segment) for segment in range(SEGMENTS)]


def test_quad_loops_are_the_rings_between_the_caps():
    mesh = capped_cylinder(triangles=False)
    everywhere = numpy.ones(len(mesh[2]), dtype=bool)

    loops = EdgeLoops(*mesh).quad_loops(loop_region(mesh, everywhere))

    assert all(len({vertex // SEGMENTS for vertex in loop}) == 1 for loop in loops)
    assert all(len(loop) == SEGMENTS for loop in loops)
    # the rings next to the caps touch triangles
    assert sorted(loop[0] // SEGMENTS for loop in loops) == list(range(1, RINGS - 1))


def test_drawn_loops_skip_the_rings_that_cut_off_only_a_flat_cap():
    mesh = capped_cylinder(triangles=False, ring_spacing=0.1)
    everywhere = numpy.ones(len(mesh[2]), dtype=bool)

    loops = drawn_loops(EdgeLoops(*mesh), loop_region(mesh, everywhere))

    # rings 1 and 2 cut off a nearly flat cap, which scores 1.2 and 1.4
    assert sorted(loop[0] // SEGMENTS for loop in loops) == list(range(3, RINGS - 3))


def test_drawn_loops_keep_rings_whose_small_side_reaches_the_region_border():
    mesh = capped_cylinder(triangles=False, ring_spacing=0.1)
    in_region = numpy.ones(len(mesh[2]), dtype=bool)
    # capped_cylinder lists the cap triangles after the quads, bottom then top
    in_region[(RINGS - 1) * SEGMENTS :: 2] = False

    loops = drawn_loops(EdgeLoops(*mesh), loop_region(mesh, in_region))

    assert sorted(loop[0] // SEGMENTS for loop in loops) == list(range(1, RINGS - 3))


def test_drawn_loops_on_triangles_go_around_the_cylinder():
    mesh = capped_cylinder(triangles=True, jitter=0.03)
    everywhere = numpy.ones(len(mesh[2]), dtype=bool)

    loops = drawn_loops(EdgeLoops(*mesh), loop_region(mesh, everywhere))

    assert loops
    every_segment = set(range(SEGMENTS))
    assert all(
        {vertex % SEGMENTS for vertex in loop} == every_segment for loop in loops
    )
    assert sum(len(loop) for loop in loops) == len(set().union(*loops))


def test_untouching_loops_keep_both_ends_of_a_chain_over_its_shorter_middle():
    first, middle, last = [0, 1, 2], [2, 3, 4], [4, 5, 6]

    kept = untouching_loops([first, middle, last], [2.0, 1.0, 2.0])

    assert sorted(kept) == [first, last]


def test_untouching_loops_keep_the_shortest_when_all_touch():
    loops = [[0, 1, 2], [0, 3, 4], [0, 5, 6]]

    assert untouching_loops(loops, [3.0, 1.0, 2.0]) == [[0, 3, 4]]


def test_a_diagonal_triangle_edge_finds_the_ring_on_a_jittered_cylinder():
    mesh = capped_cylinder(triangles=True, jitter=0.03)
    everywhere = numpy.ones(len(mesh[2]), dtype=bool)
    # a diagonal of the quad between rings five and six
    tail, head = ring_vertex(5, 0), ring_vertex(6, 1)

    loop = loop_through(mesh, everywhere, tail, head)

    assert len(loop) == SEGMENTS
    assert len({vertex // SEGMENTS for vertex in loop}) <= 2


def test_loops_split_the_region_in_two_and_stay_inside_it():
    mesh = capped_cylinder(triangles=False)
    face_count = len(mesh[2])
    upper = numpy.zeros(face_count, dtype=bool)
    # quads come ring by ring, then the caps alternate bottom and top
    upper[(RINGS // 2) * SEGMENTS : (RINGS - 1) * SEGMENTS] = True
    upper[(RINGS - 1) * SEGMENTS + 1 :: 2] = True

    ring = loop_through(mesh, upper, ring_vertex(8, 3), ring_vertex(8, 4))
    lengthwise = loop_through(mesh, upper, ring_vertex(8, 3), ring_vertex(9, 3))

    side = loop_side(loop_region(mesh, upper), ring)
    assert sorted(ring) == [ring_vertex(8, segment) for segment in range(SEGMENTS)]
    assert len(side) in (2 * SEGMENTS, upper.sum() - 2 * SEGMENTS)
    assert upper[side].all()
    # the quad walk along the length leaves the region
    assert lengthwise is None
