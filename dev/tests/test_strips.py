import collections
import importlib.util
import math
from pathlib import Path

# loaded from file so it doesn't need the bpy-only addon package
spec = importlib.util.spec_from_file_location(
    "strips", Path(__file__).parents[2] / "src" / "strips.py"
)
strips = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strips)

FIXTURES = Path(__file__).parent / "fixtures"


def read_obj(path):
    verts, faces = [], []
    for line in open(path):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return verts, faces


def region_count(verts, faces, width):
    weighted, areas, edges = strips.build(verts, faces)
    find = strips.partition(faces, weighted, edges, strips.LOW_ANGLE)
    if width == "auto":
        width = strips.detect_width(
            verts, faces, areas, edges, find, strips.diagonal(verts)
        )
    label, _ = strips.absorb(verts, faces, weighted, areas, edges, find, width)
    return len(set(label.values()))


def test_beveled_cube_merges_to_faces():
    # 2-segment bevel: no per-edge angle can give 6 charts, strip merging must
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    assert region_count(verts, faces, "auto") == 6


def test_beveled_cube_zero_width_keeps_bands():
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    assert region_count(verts, faces, 0.0) == 54


def test_plain_cube_is_untouched():
    # no narrow regions, so auto width must not merge the 6 faces
    verts, faces = read_obj(FIXTURES / "cube.obj")
    assert region_count(verts, faces, "auto") == 6


def test_seam_edges_on_plain_cube():
    verts, faces = read_obj(FIXTURES / "cube.obj")
    seams = strips.seam_edges(verts, faces)
    # every one of the 12 cube edges separates two faces, the triangulation
    # diagonals stay interior
    assert len(seams) == 12
    assert all(v0 < v1 for v0, v1 in seams)


def tube(sides=4, height=1.0):
    """Open-ended tube: merging its way around the ring would make an annulus."""
    verts, faces = [], []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        verts.append([math.cos(angle), math.sin(angle), 0.0])
        verts.append([math.cos(angle), math.sin(angle), height])
    for i in range(sides):
        low, high = 2 * i, 2 * i + 1
        next_low, next_high = 2 * ((i + 1) % sides), 2 * ((i + 1) % sides) + 1
        faces.append([low, next_low, high])
        faces.append([next_low, next_high, high])
    return verts, faces


def test_tube_stops_merging_before_it_closes_the_ring():
    verts, faces = tube()
    weighted, areas, edges = strips.build(verts, faces)
    find = strips.partition(faces, weighted, edges, strips.LOW_ANGLE)
    # width far over the tube's own, so every side is a merge candidate
    label, _ = strips.absorb(verts, faces, weighted, areas, edges, find, 100.0)
    ec, _, _ = strips.region_topology(edges, label)
    assert len(set(label.values())) == 2
    assert all(value == 1 for value in ec.values())


def test_coarse_tube_merges_into_two_halves():
    # 22.5 degrees a segment, so the partition shatters the wall into 16
    # columns and no flatness test can put it back: the smooth merge reads the
    # turn at each boundary instead and takes the wall to two halves. Closing
    # the ring is refused too: at height 1 the cut-open wall would unroll to
    # aspect 2*pi, past the strip bound
    verts, faces = tube(16)
    seams = strips.seam_edges(verts, faces)
    assert len(seams) == 2
    assert all({verts[v0][2], verts[v1][2]} == {0.0, 1.0} for v0, v1 in seams)


def test_tall_coarse_tube_closes_ring_for_one_cut():
    # same wall at height 2 unrolls to aspect pi, so close_rings merges the
    # halves back into an annulus and disk_cuts opens it with a single cut
    verts, faces = tube(16, height=2.0)
    seams = strips.seam_edges(verts, faces)
    assert len(seams) == 1
    ((v0, v1),) = seams
    assert {verts[v0][2], verts[v1][2]} == {0.0, 2.0}


def test_ring_closing_refuses_a_crease():
    # two halves of an octagonal prism pass the aspect bound, but their
    # boundaries turn 45 degrees: creases, so the ring must stay open
    verts, faces = tube(8, height=2.0)
    weighted, areas, edges = strips.build(verts, faces)
    label = {i: 0 if i < 8 else 1 for i in range(len(faces))}
    assert strips.close_rings(verts, weighted, areas, edges, label) == label


def test_faceted_tube_keeps_its_facets():
    # 45 degrees a segment reads as a crease, and an octagonal prism is one
    verts, faces = tube(8)
    assert len(strips.seam_edges(verts, faces)) == 8


def test_seamless_closed_mesh_retries_at_the_floor():
    # a hex head smears every feature to just under 60, so at 66 close_rings
    # seals the closed mesh into one seamless region nothing can flatten:
    # detection must fall back to CREASE_ANGLE instead of returning nothing
    verts, faces = read_obj(
        Path(__file__).parents[1] / "bench/models/hard-surface/sharp/fastener_03.obj"
    )
    faces = [tuple(f) for f in faces]
    at_floor = strips.seam_edges(verts, faces, strips.CREASE_ANGLE)
    assert at_floor
    assert strips.seam_edges(verts, faces, 66) == at_floor


def test_lower_feature_angle_keeps_shallow_seams():
    # 22.5 degree panel boundaries merge away at the default 30 but survive
    # 15: the knob's point, more seams toward artist style
    verts, faces = tube(16)
    assert len(strips.seam_edges(verts, faces, angle=15)) > len(
        strips.seam_edges(verts, faces)
    )


def test_beveled_cube_keeps_six_charts_through_every_merge():
    # the smooth merge sees 22.5 degrees across what is left of a dissolved
    # bevel, so without absorb handing over the turn it carried this collapses
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    seams = strips.seam_edges(verts, faces)
    edges = strips.face_edges(faces)
    assert len(strips.island_groups(faces, seams, edges)) == 6


def test_absorbed_bevel_still_reads_as_a_crease():
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    weighted, areas, edges = strips.build(verts, faces)
    find = strips.partition(faces, weighted, edges, strips.LOW_ANGLE)
    width = strips.detect_width(
        verts, faces, areas, edges, find, strips.diagonal(verts)
    )
    _, bounds = strips.absorb(verts, faces, weighted, areas, edges, find, width)
    live = [key for key in bounds.length if bounds.length[key] > 0]
    # every boundary left is a cube corner with its bevel absorbed into one
    # side, so all of them must still turn the full 90 degrees. A corner patch
    # carries two bevels at once, which reads a little over
    angles = [bounds.turn[key] / bounds.length[key] for key in live]
    assert angles
    assert min(angles) > strips.CREASE_ANGLE
    assert max(angles) < 120
    # the turn at the boundary's own edges is only the last bevel segment, so
    # the carry is what the crease reading rests on here
    assert min(bounds.step[key] / bounds.length[key] for key in live) < 45
    # and it is spread over a bevel's width, not a surface's, which is what
    # keeps the smooth merge from reading it as curvature
    assert max(bounds.spread[key] / bounds.length[key] for key in live) < width


def capped_tube(sides=32, height=2.0):
    """A tube with a flat triangle-fan cap on top: one region of it is the
    smooth-model sock, a wall merged over its end cap."""
    verts, faces = tube(sides, height)
    top = len(verts)
    verts.append([0.0, 0.0, height])
    for i in range(sides):
        faces.append([2 * i + 1, 2 * ((i + 1) % sides) + 1, top])
    return verts, faces


def elbow(rings=12, sides=12, bend_radius=3.0, tube_radius=1.0):
    """A quarter-torus tube: swept, but around a bending axis."""
    verts, faces = [], []
    for i in range(rings + 1):
        t = (math.pi / 2) * i / rings
        for j in range(sides):
            p = 2 * math.pi * j / sides
            spoke = bend_radius + tube_radius * math.cos(p)
            verts.append(
                [spoke * math.cos(t), tube_radius * math.sin(p), spoke * math.sin(t)]
            )
    for i in range(rings):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


def sweep_regions(verts, faces):
    weighted, areas, edges = strips.build(verts, faces)
    label = {i: 0 for i in range(len(faces))}
    return strips.split_sweeps(weighted, areas, edges, label), len(faces)


def test_sweep_split_lifts_the_cap_off_a_sock():
    verts, faces = capped_tube()
    label, count = sweep_regions(verts, faces)
    regions = collections.defaultdict(set)
    for i, r in label.items():
        regions[r].add(i)
    assert len(regions) == 2
    # the cap fan is exactly the faces added after the wall quads
    fan = set(range(count - 32, count))
    assert fan in regions.values()


def test_sweep_split_leaves_a_bent_tube_alone():
    # mid-bend normals sit between wall and cap against any axis, so an
    # elbow must not read as a sock: it unrolls fine once cut open
    verts, faces = elbow()
    label, _ = sweep_regions(verts, faces)
    assert len(set(label.values())) == 1


def filleted_tube(sides=32, height=2.0, fillet=8, radius=0.4):
    """A tube whose flat cap meets the wall through a rounded fillet, so no
    boundary in it turns like a crease: the smooth-model sock."""
    verts, faces = [], []
    rings = [(1.0, 0.0), (1.0, height)]
    for k in range(1, fillet + 1):
        t = (math.pi / 2) * k / fillet
        rings.append((1 - radius * (1 - math.cos(t)), height + radius * math.sin(t)))
    for r, z in rings:
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            verts.append([r * math.cos(angle), r * math.sin(angle), z])
    for ring in range(len(rings) - 1):
        for i in range(sides):
            a = ring * sides + i
            b = ring * sides + (i + 1) % sides
            c = (ring + 1) * sides + (i + 1) % sides
            d = (ring + 1) * sides + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    top = len(verts)
    verts.append([0.0, 0.0, rings[-1][1]])
    last = (len(rings) - 1) * sides
    for i in range(sides):
        faces.append([last + i, last + (i + 1) % sides, top])
    return verts, faces


def test_filleted_cap_is_cut_off_at_its_rim():
    # the wall merges straight over a filleted cap, no boundary turns like a
    # crease, so only the sweep split separates them
    verts, faces = filleted_tube()
    edges = strips.face_edges(faces)
    with_rims = strips.seam_edges(verts, faces)
    without = strips.seam_edges(verts, faces, rims=False)
    assert len(strips.island_groups(faces, without, edges)) == 1
    assert len(strips.island_groups(faces, with_rims, edges)) == 2


def test_sweep_split_leaves_a_shallow_shell_alone():
    # a quarter of the capped tube: the wall barely turns, so this is a
    # curved plate with a flange, not a sock, and rims make no sense on it
    verts, faces = capped_tube()
    quarter = faces[0:16] + faces[64:72]
    label, _ = sweep_regions(verts, quarter)
    assert len(set(label.values())) == 1


def euler_after_cut(ec, cuts):
    """Slitting a region along a boundary-to-boundary path splits every vertex
    on it and every one of its edges, so EC goes up by one per cut."""
    return ec + len({v for edge in cuts for v in edge}) - len(cuts)


def test_annulus_region_is_cut_open():
    # the whole tube as one region, which is what a low-curvature tube wall
    # partitions into and no merge can repair
    verts, faces = tube(8)
    _, _, edges = strips.build(verts, faces)
    label = {i: 0 for i in range(len(faces))}
    ec, _, _ = strips.region_topology(edges, label)
    assert ec[0] == 0

    cuts = strips.disk_cuts(verts, edges, label)
    assert len(cuts) == 1  # one rim to the other, along a single side edge
    v0, v1 = next(iter(cuts))
    assert {verts[v0][2], verts[v1][2]} == {0.0, 1.0}
    assert euler_after_cut(ec[0], cuts) == 1


def folded_pair(z):
    """Two triangles sharing the edge (0, 1), the second lifted to z."""
    verts = [[0, 0, 0], [0, 1, 0], [-1, 0.5, 0], [1, 0.5, z]]
    faces = [[0, 1, 2], [1, 0, 3]]
    return verts, faces


def test_crease_relief_orders_concave_convex_flat():
    def relief_at(z):
        verts, faces = folded_pair(z)
        weighted, _, edges = strips.build(verts, faces)
        relief = strips.crease_relief(verts, faces, weighted, edges)
        return relief.get(strips.pair(0, 1), 1.0)

    concave, convex, flat = relief_at(1.0), relief_at(-1.0), relief_at(0.0)
    assert concave < convex < flat == 1.0


def folded_flap():
    """Two flat quad columns and a third folded straight up at x=1, so the
    fold line is the only crease."""
    rows = 7
    verts = (
        [[0, y, 0] for y in range(rows)]
        + [[1, y, 0] for y in range(rows)]
        + [[1, y, 1] for y in range(rows)]
    )
    faces = []
    for y in range(rows - 1):
        faces.append([y, rows + y, rows + y + 1, y + 1])
        faces.append([rows + y, 2 * rows + y, 2 * rows + y + 1, rows + y + 1])
    return verts, faces


def test_cut_path_rides_the_crease():
    # the fold route is 8 long against 6 direct, so only the crease
    # discount can make it win
    verts, faces = folded_flap()
    weighted, _, edges = strips.build(verts, faces)
    relief = strips.crease_relief(verts, faces, weighted, edges)
    adjacent = vertex_adjacency(faces)

    plain = strips.cut_path(verts, adjacent, {0}, {6})
    assert all(verts[v][0] == 0 for v in plain)

    creased = strips.cut_path(verts, adjacent, {0}, {6}, relief=relief)
    assert sum(1 for v in creased if verts[v][0] == 1 and verts[v][2] == 0) == 7


def rook_grid():
    """3x3 vertex grid with only horizontal and vertical edges, so every
    corner-to-corner path is 4 long and only turn count tells them apart."""
    verts = [(x, y, 0.0) for y in range(3) for x in range(3)]
    adjacent = collections.defaultdict(set)
    for y in range(3):
        for x in range(3):
            a = 3 * y + x
            if x < 2:
                adjacent[a].add(a + 1)
                adjacent[a + 1].add(a)
            if y < 2:
                adjacent[a].add(a + 3)
                adjacent[a + 3].add(a)
    return verts, adjacent


def turn_count(verts, path):
    turns = 0
    for u, v, w in zip(path, path[1:], path[2:]):
        a = [verts[v][i] - verts[u][i] for i in range(3)]
        b = [verts[w][i] - verts[v][i] for i in range(3)]
        if strips.cross(a, b) != [0, 0, 0]:
            turns += 1
    return turns


def test_dull_cut_is_a_line():
    # every corner-to-corner path is 4 long, so without the turn penalty the
    # staircase can win on heap order; with it the single-corner L must
    verts, adjacent = rook_grid()
    path = strips.cut_path(verts, adjacent, {0}, {8}, relief={})
    assert len(path) == 5
    assert turn_count(verts, path) == 1


def test_creased_staircase_beats_the_line():
    # the same grid with a creased staircase: crease edges are exempt from
    # the turn penalty and discounted, so the seam follows the crease
    verts, adjacent = rook_grid()
    stairs = [(0, 1), (1, 4), (4, 5), (5, 8)]
    relief = {strips.pair(a, b): 0.85 for a, b in stairs}
    path = strips.cut_path(verts, adjacent, {0}, {8}, relief=relief)
    assert set(path) == {0, 1, 4, 5, 8}


def test_path_cost_prices_turns():
    verts, _ = rook_grid()
    straight = strips.path_cost(verts, [0, 1, 2], relief={})
    bent = strips.path_cost(verts, [0, 1, 4], relief={})
    assert straight == 2.0
    assert bent == 2.0 + strips.TURN_COST * 0.5


def flat_grid():
    """2x2 quad grid of 8 triangles on z=0, split down the middle line."""
    verts = [[x, y, 0] for y in range(3) for x in range(3)]
    faces = []
    for y in range(2):
        for x in range(2):
            a = 3 * y + x
            faces.append([a, a + 1, a + 4])
            faces.append([a, a + 4, a + 3])
    return verts, faces


def test_tooth_on_a_flat_boundary_is_flattened():
    verts, faces = flat_grid()
    weighted, _, edges = strips.build(verts, faces)
    # left column region 0, right column region 1, except one right triangle
    # sticking into the left as a tooth
    label = {0: 0, 1: 0, 4: 0, 5: 0, 2: 1, 3: 1, 6: 1, 7: 1}
    label[3] = 0

    flat = strips.flatten_teeth(weighted, faces, edges, label)
    assert flat[3] == 1
    assert strips.boundary_edges(edges, flat) == {(1, 4), (4, 7)}


def folded_planes():
    """Two quad planes meeting at 90 degrees along the line y=0."""
    verts = (
        [[x, 0, 0] for x in range(3)]
        + [[x, -1, 0] for x in range(3)]
        + [[x, 0, 1] for x in range(3)]
    )
    faces = []
    for x in range(2):
        faces.append([x, x + 1, x + 4])
        faces.append([x, x + 4, x + 3])
    for x in range(2):
        faces.append([x, x + 1, x + 7])
        faces.append([x, x + 7, x + 6])
    return verts, faces


def test_tooth_flip_returns_the_seam_to_the_fold():
    verts, faces = folded_planes()
    weighted, _, edges = strips.build(verts, faces)
    # one vertical triangle mislabeled onto the flat plane: its kept edge is
    # the fold itself, so the flip wins even though its lost edges are dull
    label = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    label[4] = 0

    flat = strips.flatten_teeth(weighted, faces, edges, label)
    assert flat[4] == 1
    assert strips.boundary_edges(edges, flat) == {(0, 1), (1, 2)}


def test_corner_on_a_crease_is_not_cut():
    verts, faces = read_obj(FIXTURES / "cube.obj")
    weighted, _, edges = strips.build(verts, faces)
    # top face one region, the rest another: each top triangle has two sharp
    # boundary edges and only a dull diagonal to keep, so no flip may round
    # the corner off
    top = max(v[2] for v in verts)
    label = {
        f: 0 if all(verts[v][2] == top for v in face) else 1
        for f, face in enumerate(faces)
    }

    assert strips.flatten_teeth(weighted, faces, edges, label) == label


def bump_grid():
    """3x3 quad grid on z=0 split down x=1, with the middle right quad
    labeled across the line so the boundary detours around it."""
    verts = [[x, y, 0] for y in range(4) for x in range(4)]
    faces = []
    for y in range(3):
        for x in range(3):
            a = 4 * y + x
            faces.append([a, a + 1, a + 5])
            faces.append([a, a + 5, a + 4])
    label = {f: 0 if (f // 2) % 3 == 0 else 1 for f in range(18)}
    label[8] = label[9] = 0
    return verts, faces, label


def test_reroute_straightens_a_boundary_bump():
    verts, faces, label = bump_grid()
    weighted, areas, edges = strips.build(verts, faces)
    relief = strips.crease_relief(verts, faces, weighted, edges)

    moved = strips.reroute_boundaries(verts, faces, areas, edges, label, relief)
    assert moved[8] == 1 and moved[9] == 1
    assert strips.boundary_edges(edges, moved) == {(1, 5), (5, 9), (9, 13)}


def walled_floor():
    """A floor with a wall folded up along y=0."""
    verts = [[x, y, 0] for y in range(3) for x in range(9)] + [
        [x, 0, 1] for x in range(9)
    ]
    faces = []
    for y in range(2):
        for x in range(8):
            a = 9 * y + x
            faces.append([a, a + 1, a + 10])
            faces.append([a, a + 10, a + 9])
    for x in range(8):
        faces.append([x, x + 1, 28 + x])
        faces.append([x, 28 + x, 27 + x])
    return verts, faces


def test_reroute_drops_the_seam_onto_the_fold():
    verts, faces = walled_floor()
    # the wall region reaches over the fold onto the middle of the floor, so
    # the seam runs mostly on flat ground: only the relief discount along the
    # fold can pay for the longer straight route
    label = {f: 1 if f >= 32 else 0 for f in range(48)}
    for f in range(4, 12):
        label[f] = 1
    weighted, areas, edges = strips.build(verts, faces)
    relief = strips.crease_relief(verts, faces, weighted, edges)

    moved = strips.reroute_boundaries(verts, faces, areas, edges, label, relief)
    assert all(moved[f] == (1 if f >= 32 else 0) for f in range(48))
    assert strips.boundary_edges(edges, moved) == {(x, x + 1) for x in range(8)}


def capped_prism(sides=12):
    """Open-bottomed prism with a fan-triangulated top cap and a sharp rim."""
    verts = [[0.0, 0.0, 1.0]]
    for ring_z in (1.0, 0.0):
        for i in range(sides):
            a = 2 * math.pi * i / sides
            verts.append([math.cos(a), math.sin(a), ring_z])
    faces = []
    for i in range(sides):
        faces.append([0, 1 + i, 1 + (i + 1) % sides])
    for i in range(sides):
        a, b = 1 + i, 1 + (i + 1) % sides
        faces.append([b, a, sides + a])
        faces.append([b, sides + a, sides + b])
    return verts, faces


def test_reroute_snaps_a_closed_loop_to_its_rim():
    verts, faces = capped_prism()
    # one cap triangle labeled into the wall, so the rim loop detours over
    # the cap through its center. No junction anchors the loop, only the
    # loop handling can pull it back
    label = {f: 0 if f < 12 else 1 for f in range(36)}
    label[0] = 1
    weighted, areas, edges = strips.build(verts, faces)
    relief = strips.crease_relief(verts, faces, weighted, edges)

    moved = strips.reroute_boundaries(verts, faces, areas, edges, label, relief)
    assert moved[0] == 0
    rim = {strips.pair(1 + i, 1 + (i + 1) % 12) for i in range(12)}
    assert strips.boundary_edges(edges, moved) == rim


def test_disk_regions_are_left_alone():
    verts, faces = read_obj(FIXTURES / "cube.obj")
    weighted, areas, edges = strips.build(verts, faces)
    find = strips.partition(faces, weighted, edges, strips.LOW_ANGLE)
    label = {i: find(i) for i in range(len(faces))}
    assert strips.disk_cuts(verts, edges, label) == set()


def strip_island(quads, scale=1.0, angle=0.0):
    """A strip of unit quads laid out flat in uv, as the verts, faces and
    per-face corner uvs split_islands takes. Verts sit in the z=0 plane so 3d
    path lengths match uv lengths."""
    points, faces = [], []
    cos, sin = math.cos(angle), math.sin(angle)
    for i in range(quads + 1):
        for y in (0.0, 1.0):
            x = float(i)
            points.append((scale * (x * cos - y * sin), scale * (x * sin + y * cos)))
    for i in range(quads):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces.append([a, c, b])
        faces.append([c, d, b])
    uvs = [[points[v] for v in f] for f in faces]
    verts = [(x, y, 0.0) for x, y in points]
    return verts, faces, uvs


def fold_face(uvs, fi):
    """Swap two corner uvs so the face inverts in uv, like a SLIM fold."""
    uvs[fi] = [uvs[fi][1], uvs[fi][0], uvs[fi][2]]


def island_count(faces, seams):
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            owners.setdefault(strips.pair(f[i], f[(i + 1) % len(f)]), []).append(fi)
    for key, fs in owners.items():
        if len(fs) == 2 and key not in seams:
            a, b = find(fs[0]), find(fs[1])
            if a != b:
                parent[a] = b
    return len({find(i) for i in range(len(faces))})


def test_clean_long_island_splits_for_packing():
    # flip-free but a 30:1 strip across the atlas packs badly, so it splits
    # like a ruined strip would
    verts, faces, uvs = strip_island(30)
    extra = strips.split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 5


def test_clean_short_strip_is_not_split():
    # same shape at 1/200 scale: thin but small, the kind a bevel band
    # leaves, and cutting those would shatter every beveled model
    verts, faces, uvs = strip_island(30, scale=1 / 200)
    assert strips.split_islands(verts, faces, set(), uvs) == set()


def test_clean_compact_island_is_not_split():
    verts, faces, uvs = strip_island(4)
    assert strips.split_islands(verts, faces, set(), uvs) == set()


def test_folded_long_island_is_split_into_even_pieces():
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 20)
    extra = strips.split_islands(verts, faces, set(), uvs)
    assert extra
    # aspect ~29, so 5 slices, each just under the bound
    assert island_count(faces, extra) == 5


def test_split_is_rotation_invariant():
    verts, faces, uvs = strip_island(30, angle=0.7)
    fold_face(uvs, 20)
    extra = strips.split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 5


def test_folded_compact_island_is_halved():
    # not a strip, but ruined is ruined: the engine would re-cut it anyway,
    # so it gets one cut and each half a fresh chance to unwrap clean
    verts, faces, uvs = strip_island(4)
    fold_face(uvs, 2)
    extra = strips.split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 2


def test_short_ruined_island_is_still_halved():
    # same folded 30:1 shape at 1/200 scale: 0.15 of the atlas, under
    # SPLIT_LENGTH, so it is halved instead of sliced into aspect bins
    verts, faces, uvs = strip_island(30, scale=1 / 200)
    fold_face(uvs, 20)
    extra = strips.split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 2


def test_halving_cut_takes_the_shortest_path():
    # 5 quads put the halving line mid-quad, where the raw bin cut takes the
    # sqrt(2) diagonal: straightening must slide a face across so the cut
    # lands on a unit column edge, either side of the line
    verts, faces, uvs = strip_island(5)
    fold_face(uvs, 2)
    extra = strips.split_islands(verts, faces, set(), uvs)
    assert extra in ({strips.pair(4, 5)}, {strips.pair(6, 7)})


def test_split_scan_restricted_to_given_groups():
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 20)
    assert strips.split_islands(verts, faces, set(), uvs, None, []) == set()
    everything = [list(range(len(faces)))]
    assert strips.split_islands(verts, faces, set(), uvs, None, everything)


def test_split_respects_existing_seams():
    # a seam already cuts the strip in half, so each folded island measures
    # 15:1 and splits into 3, and the seam edge itself must not come back
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 10)
    fold_face(uvs, 50)
    mid = strips.pair(30, 31)
    extra = strips.split_islands(verts, faces, {mid}, uvs)
    assert mid not in extra
    assert island_count(faces, extra | {mid}) == 6


def test_uv_islands_follow_the_uv_map():
    # a flat strip is one island until its uvs split mid-column, no seam
    # marks involved
    verts, faces, uvs = strip_island(4)
    edges = strips.face_edges(faces)
    assert len(strips.uv_island_groups(faces, uvs, edges)) == 1
    for fi in range(4, 8):
        uvs[fi] = [(u + 5.0, v) for u, v in uvs[fi]]
    groups = strips.uv_island_groups(faces, uvs, edges)
    assert sorted(len(g) for g in groups) == [4, 4]


def vertex_adjacency(faces):
    adjacent = collections.defaultdict(set)
    for a, b in strips.face_edges(faces):
        adjacent[a].add(b)
        adjacent[b].add(a)
    return adjacent


def test_snap_paths_redraws_a_cut_on_the_dense_mesh():
    # a two segment cut whose three corners land on grid vertices comes back
    # as one connected run of real edges
    verts, faces, _ = grid_island(4, 4)
    mapped = [0, 2, 12]
    paths = strips.snap_paths(verts, vertex_adjacency(faces), mapped, {(0, 1), (1, 2)})
    assert paths == {(0, 1), (1, 2), (2, 7), (7, 12)}


def test_snap_paths_drops_a_cut_with_nowhere_to_go():
    # both ends on one vertex, then an end nothing connects to
    verts, faces, _ = grid_island(4, 4)
    adjacent = vertex_adjacency(faces)
    mapped = [3, 3, len(verts)]
    assert strips.snap_paths(verts, adjacent, mapped, {(0, 1)}) == set()
    assert strips.snap_paths(verts, adjacent, mapped, {(0, 2)}) == set()


def test_uv_fit_scales_into_old_bounds():
    move = strips.uv_fit([(0, 0), (2, 1)], (10, 10, 11, 10.5))
    assert move((0, 0)) == (10, 10)
    assert move((2, 1)) == (11, 10.5)
    # aspect mismatch keeps the scale uniform and the island inside
    move = strips.uv_fit([(0, 0), (2, 1)], (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0.25)
    assert move((2, 1)) == (1, 0.75)


def test_uv_area_fit_keeps_the_old_uv_area():
    # two charts packed in a square, the island they came from was a 4:1 strip
    charts = [
        [(0, 0), (1, 0), (1, 0.4), (0, 0.4)],
        [(0, 0.6), (1, 0.6), (1, 1), (0, 1)],
    ]
    move = strips.uv_area_fit(charts, 0.8, (0, 0, 0.8, 0.2))
    moved = [[move(uv) for uv in chart] for chart in charts]
    area = sum(abs(strips.signed_area(chart)) for chart in moved)
    assert math.isclose(area, 0.8)
    # centered on the old spot, and a bbox fit would have shrunk it instead
    xs = [u for chart in moved for u, _ in chart]
    ys = [v for chart in moved for _, v in chart]
    assert math.isclose((min(xs) + max(xs)) / 2, 0.4)
    assert math.isclose((min(ys) + max(ys)) / 2, 0.1)
    assert max(ys) - min(ys) > 0.2


def test_uv_area_fit_falls_back_to_the_bbox_without_an_area():
    square = [[(0, 0), (2, 0), (2, 2), (0, 2)]]
    move = strips.uv_area_fit(square, 0, (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0)
    assert move((2, 2)) == (1, 1)


def grid_island(cols, rows):
    """A triangulated vertex grid laid out flat in uv, one island."""
    verts = [
        (float(x), float(y), 0.0) for y in range(rows + 1) for x in range(cols + 1)
    ]
    faces = []
    for cy in range(rows):
        for cx in range(cols):
            a = cy * (cols + 1) + cx
            b, c, d = a + 1, a + cols + 2, a + cols + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    return verts, faces, uvs


def annulus_island():
    """A flat 3x3 quad ring with the middle cell missing, identity uvs: no
    flips or crossings, ruined by topology alone."""
    verts, faces, uvs = grid_island(3, 3)
    hole = [faces.index([5, 6, 10]), faces.index([5, 10, 9])]
    for fi in sorted(hole, reverse=True):
        del faces[fi]
        del uvs[fi]
    return verts, faces, uvs


def test_uv_topology_reads_the_unwrap_not_the_mesh():
    verts, faces, uvs = strip_island(4)
    edges = strips.face_edges(faces)
    ec, loops = strips.uv_topology(list(range(len(faces))), faces, edges, set())
    assert ec == 1 and len(loops) == 1
    # a seam splits corners apart: still one disk per side of the cut
    seams = {strips.pair(4, 5)}
    for group in strips.island_groups(faces, seams, edges):
        ec, loops = strips.uv_topology(group, faces, edges, seams)
        assert ec == 1 and len(loops) == 1


def test_annulus_island_is_ruined_and_opened_not_split():
    verts, faces, uvs = annulus_island()
    edges = strips.face_edges(faces)
    group = list(range(len(faces)))
    ec, loops = strips.uv_topology(group, faces, edges, set())
    assert ec == 0 and len(loops) == 2
    assert strips.island_ruined(group, faces, uvs, edges, set())

    extra = strips.split_islands(verts, faces, set(), uvs)
    assert extra
    # opened, not split: still one island, and a disk once the cut is a seam
    assert island_count(faces, extra) == 1
    ec, loops = strips.uv_topology(group, faces, edges, extra)
    assert ec == 1 and len(loops) == 1


def test_slit_sides_crossing_counts_as_ruined():
    # a dangling seam between two interior verts makes a slit whose sides
    # share both mesh verts, and once the sides separate in uv they can
    # cross like any other boundary pair
    verts, faces, uvs = grid_island(3, 2)
    edges = strips.face_edges(faces)
    seams = {strips.pair(5, 6)}
    group = list(range(len(faces)))
    below = faces.index([1, 6, 5])
    above = faces.index([5, 6, 10])
    assert not strips.island_ruined(group, faces, uvs, edges, seams)
    uvs[below] = [(1.0, 0.0), (1.8, 0.85), (1.2, 0.85)]
    uvs[above] = [(1.4, 0.7), (1.6, 1.0), (2.0, 2.0)]
    assert strips.island_ruined(group, faces, uvs, edges, seams)


def test_crossing_boundary_counts_as_ruined():
    # collinear overlap included, the branch a naive segment test misses
    assert strips.crosses((0, 0), (2, 0), (1, -1), (1, 1))
    assert not strips.crosses((0, 0), (2, 0), (0, 1), (2, 1))
    assert strips.crosses((0, 0), (2, 0), (3, 0), (1, 0))
    assert not strips.crosses((0, 0), (2, 0), (3, 0), (5, 0))


def test_vertex_components_join_on_a_shared_vertex():
    verts, faces = tube(sides=4)
    apart_verts, apart_faces = tube(sides=4)
    offset = len(verts)
    verts += [[v[0] + 5.0, v[1], v[2]] for v in apart_verts]
    faces += [[i + offset for i in f] for f in apart_faces]
    comps = strips.vertex_components(faces)
    assert sorted(len(c) for c in comps) == [8, 8]
    # welding one vertex joins them, the connectivity mesh.separate uses
    welded = [[0 if i == offset else i for i in f] for f in faces]
    assert len(strips.vertex_components(welded)) == 1
