from turtle import shape
import taichi as ti

# node_construct()
# "node_basic" -> basic
def struct_node_basic(dim, node_num):
    struct_node_basic = ti.types.struct(
        pos=ti.types.vector(dim, ti.f32),
        vel=ti.types.vector(dim, ti.f32),
        acc=ti.types.vector(dim, ti.f32),
        mass=ti.f32,
        rest_density=ti.f32,
        rest_volume=ti.f32,
        size=ti.f32
    )
    return struct_node_basic.field(shape=(node_num,))

# node_construct()
# "node_implicit_sph" -> implicit_sph
def struct_node_implicit_sph(dim, node_num):
    struct_node_implicit_sph = ti.types.struct(
        W=ti.f32,
        W_grad=ti.types.vector(dim, ti.f32),

        alpha_1=ti.types.vector(dim, ti.f32),
        alpha_2=ti.f32,

        vel_adv=ti.types.vector(dim, ti.f32),
        acce_adv=ti.types.vector(dim, ti.f32),

        approximated_compression_ratio=ti.f32,
        approximated_density=ti.f32,
        approximated_compression_ratio_adv=ti.f32,
        approximated_density_adv=ti.f32,
    )
    return struct_node_implicit_sph.field(shape=(node_num,))

# node_construct()
# "node_color" -> color
def struct_node_color(node_num):
    struct_node_color = ti.types.struct(
        hex=ti.i32,
        vec=ti.types.vector(3, ti.f32),
    )
    return struct_node_color.field(shape=(node_num,))

# node_construct()
# "node_neighb_search" -> located_cell
def struct_node_neighb_search(dim, node_num):
    struct_node_neighb_search = ti.types.struct(
        vec=ti.types.vector(dim, ti.i32),
        coded=ti.i32,
        sequence=ti.i32,
        part_log=ti.i32,
    )
    return struct_node_neighb_search.field(shape=(node_num,))

# node_neighb_cell_construct()
# "node_neighb_search" -> cell
def struct_node_neighb_cell(cell_num):
    struct_node_neighb_cell = ti.types.struct(
        part_count=ti.i32,
        part_shift=ti.i32,
    )
    return struct_node_neighb_cell.field(shape=(cell_num))

