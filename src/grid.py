import bpy

GRID_IMAGE = "UVgami_UV_grid"
GRID_MATERIAL = "UVgami_grid"


def grid_material():
    if GRID_IMAGE not in bpy.data.images:
        bpy.ops.image.new(name=GRID_IMAGE, generated_type="UV_GRID")
    image = bpy.data.images.get(GRID_IMAGE)

    props = bpy.context.scene.uvgami
    if props.grid_type == "UV":
        image.generated_type = "UV_GRID"
    else:
        image.generated_type = "COLOR_GRID"
    image.generated_width = props.grid_res
    image.generated_height = props.grid_res

    material = bpy.data.materials.get(GRID_MATERIAL)
    if material is not None:
        return material

    material = bpy.data.materials.new(GRID_MATERIAL)
    # 5.0 builds the node tree here and deprecates the flag
    if bpy.app.version < (5, 0, 0):
        material.use_nodes = True
    tree_nodes = material.node_tree.nodes
    nodes = (
        tree_nodes.new(type="ShaderNodeTexImage"),
        tree_nodes.get("Principled BSDF"),
        tree_nodes.get("Material Output"),
    )
    nodes[0].image = image
    # solid texture mode draws the active image node
    tree_nodes.active = nodes[0]
    nodes[0].location = (-300, 300)
    material.node_tree.links.new(nodes[0].outputs[0], nodes[1].inputs[0])
    for node in nodes:
        node.select = False
        node.hide = True

    return material


SAVED_INDICES = "uvgami_material_indices"


def add_grid(obj, material):
    if any(slot.material is material for slot in obj.material_slots):
        return

    # rebuilding the slots drops each slot's object level material
    mesh = obj.data
    indices = [0] * len(mesh.polygons)
    mesh.polygons.foreach_get("material_index", indices)
    mesh[SAVED_INDICES] = indices

    mesh.materials.append(material)
    grid_index = len(mesh.materials) - 1
    mesh.polygons.foreach_set("material_index", [grid_index] * len(mesh.polygons))
    obj.active_material_index = grid_index


def remove_grid(obj):
    mesh = obj.data
    for material_index, material in enumerate(mesh.materials):
        if material is not None and material.name == GRID_MATERIAL:
            mesh.materials.pop(index=material_index)
            break

    indices = mesh.get(SAVED_INDICES)
    if indices is None:
        return
    del mesh[SAVED_INDICES]

    if len(indices) == len(mesh.polygons):
        mesh.polygons.foreach_set("material_index", indices)
