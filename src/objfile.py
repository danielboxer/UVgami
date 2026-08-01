def _count_elements(path):
    # obj files from blender's exporters list v/vt/vn before the first f line,
    # so stopping at the first f gives the full counts
    v = vt = vn = 0
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                v += 1
            elif line.startswith("vt "):
                vt += 1
            elif line.startswith("vn "):
                vn += 1
            elif line.startswith("f "):
                break
    return v, vt, vn


def _offset_face(line, off_v, off_vt, off_vn):
    tokens = line.split()
    new_tokens = []
    for token in tokens[1:]:
        parts = token.split("/")
        parts[0] = str(int(parts[0]) + off_v)
        if len(parts) > 1 and parts[1] != "":
            parts[1] = str(int(parts[1]) + off_vt)
        if len(parts) > 2 and parts[2] != "":
            parts[2] = str(int(parts[2]) + off_vn)
        new_tokens.append("/".join(parts))
    return "f " + " ".join(new_tokens) + "\n"


def remap_weights_to_vt(path, weights):
    """Remap v-indexed seam weights to vt indices for an obj exported with UVs.

    optcuts rebuilds a UV-carrying mesh with one vertex per vt, so the weights
    sidecar must be indexed by vt or the weights land on arbitrary vertices.
    Each vt has one source vertex (export writes one vt per vertex, uv pair)."""
    vt_to_v = {}
    with path.open() as f:
        for line in f:
            if line.startswith("f "):
                for token in line.split()[1:]:
                    parts = token.split("/")
                    if len(parts) > 1 and parts[1] != "":
                        vt_to_v.setdefault(int(parts[1]) - 1, int(parts[0]) - 1)
    return {vt: weights[v] for vt, v in sorted(vt_to_v.items()) if v in weights}


def merge_obj_files(paths):
    """Append paths[1:] into paths[0], offsetting face indices by the running
    v/vt/vn counts of the earlier files, and return paths[0]. Face tokens are
    handled generically: an obj may have v only, v/vt, or v/vt/vn."""
    off_v, off_vt, off_vn = _count_elements(paths[0])
    total_v, total_vt, total_vn = off_v, off_vt, off_vn
    with paths[0].open("a") as out:
        # since multiple obj files are combined, the size of the previous ones
        # must be added to the index numbers of the next
        for obj_path in paths[1:]:
            with obj_path.open() as f:
                for line in f:
                    if line.startswith("v "):
                        total_v += 1
                        out.write(line)
                    elif line.startswith("vt "):
                        total_vt += 1
                        out.write(line)
                    elif line.startswith("vn "):
                        total_vn += 1
                        out.write(line)
                    elif line.startswith("f "):
                        out.write(_offset_face(line, off_v, off_vt, off_vn))
                    elif line.startswith(("o ", "g ")):
                        # importer creates one object per o/g line, so drop these
                        # from appended files to keep the merged file as one object
                        pass
                    else:
                        out.write(line)
            off_v, off_vt, off_vn = total_v, total_vt, total_vn
    return paths[0]
