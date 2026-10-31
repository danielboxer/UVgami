import numpy


def loop_totals(mesh):
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    return totals


def loop_vertices_and_face_sizes(mesh):
    corners = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    return corners, loop_totals(mesh)


def vertex_position_array(mesh):
    flat = numpy.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(-1, 3)
