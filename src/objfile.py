def _element_slot(line):
    # obj face corners list their v, vt and vn indices in this order
    for slot, prefix in enumerate(("v ", "vt ", "vn ")):
        if line.startswith(prefix):
            return slot
    return None


def _count_elements(path):
    counts = [0, 0, 0]
    with path.open() as obj_file:
        for line in obj_file:
            # blender's exporters list every v/vt/vn before the first f line
            if line.startswith("f "):
                break
            slot = _element_slot(line)
            if slot is not None:
                counts[slot] += 1
    return counts


def _offset_face(line, offsets):
    corners = []
    for corner in line.split()[1:]:
        parts = corner.split("/")
        for slot, part in enumerate(parts):
            if part != "":
                parts[slot] = str(int(part) + offsets[slot])
        corners.append("/".join(parts))
    return "f " + " ".join(corners) + "\n"


def merge_obj_files(paths):
    offsets = _count_elements(paths[0])
    with paths[0].open("a") as out:
        for obj_path in paths[1:]:
            counts = [0, 0, 0]
            with obj_path.open() as obj_file:
                for line in obj_file:
                    if line.startswith("f "):
                        out.write(_offset_face(line, offsets))
                        continue
                    # Blender's obj importer creates one object per o or g line
                    if line.startswith(("o ", "g ")):
                        continue
                    out.write(line)
                    slot = _element_slot(line)
                    if slot is not None:
                        counts[slot] += 1
            offsets = [offset + count for offset, count in zip(offsets, counts)]
    return paths[0]
