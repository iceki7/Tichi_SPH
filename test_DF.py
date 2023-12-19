# runs DFSPH

import re
import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
from ti_sph.solver.DFSPH import DFSPH
import math

ti.init(arch=ti.cuda)

# /// --- CONNFIG --- ///
config_capacity = ["info_space", "info_discretization", "info_sim", "info_gui"]
config = tsph.Config(dim=3, capacity_list=config_capacity)

#initialize each value in config
# /// space ///
config_space = ti.static(config.space)
config_space.dim[None] = 3
config_space.lb[None] = [-8, -8, -8]
config_space.rt[None] = [8, 8, 8]

# /// discretization ///
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.06
config_discre.cs[None] = 220
config_discre.cfl_factor[None] = 0.5
config_discre.dt[None] = (      #zxc 固定时间步长/CFL
    tsph.fixed_dt(
        config_discre.cs[None],
        config_discre.part_size[None],
        config_discre.cfl_factor[None],
    )
    * 5
)
standart_dt = config_discre.dt[None]
config_discre.inv_dt[None] = 1 / config_discre.dt[None]

# /// sim ///
config_sim = ti.static(config.sim)
config_sim.gravity[None] = ti.Vector([0, -9.8, 0])
config_sim.kinematic_vis[None] = 1e-3

# /// gui ///
config_gui = ti.static(config.gui)
config_gui.res[None] = [1920, 1080]
config_gui.frame_rate[None] = 60
config_gui.cam_fov[None] = 55
config_gui.cam_pos[None] = [6.0, 1.0, 0.0]
config_gui.cam_look[None] = [0.0, 0.0, 0.0]
config_gui.canvas_color[None] = [0.2, 0.2, 0.6]
config_gui.ambient_light_color[None] = [0.7, 0.7, 0.7]
config_gui.point_light_pos[None] = [2, 1.5, -1.5]
config_gui.point_light_color[None] = [0.8, 0.8, 0.8]
# /// --- END OF CONNFIG --- ///

# initialize Neighb_Cell object
# /// --- NEIGHB --- ///
config_neighb = tsph.Neighb_Cell(
    dim=3,
    struct_space=config_space,
    cell_size=config_discre.part_size[None] * 2,
    search_range=1,
)
# /// --- END OF NEIGHB --- ///

# initialize Node objects (fluid and boundary) and add particles
"""""" """ OBJECT """ """"""
# /// --- INIT OBJECT --- ///
# /// FLUID ///
#zxc 粒子属性
fluid_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_implicit_sph",
    "node_neighb_search",
]
fluid = tsph.Node(
    dim=config_space.dim[None], #zxc 维度
    id=0,
    node_num=int(1e5),  
    capacity_list=fluid_capacity,#zxc 属性列表
)
fluid_node_num = fluid.push_cube_with_basic_attr(   #zxc  cube or box
    lb=ti.Vector([-1, -1.1, -1]),   #prm 区域限制
    rt=ti.Vector([1, 0.9, 1]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=1000,
    color=ti.Vector([0, 1, 1]),
)


# /// BOUND ///
bound_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_implicit_sph",
    "node_neighb_search",
]
bound = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e5),
    capacity_list=bound_capacity,
)
bound_part_num = bound.push_box_with_basic_attr(
    lb=ti.Vector([-1.5, -1.5, -1.5]),
    rt=ti.Vector([1.5, 1.5, 1.5]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    layers=3,
    rest_density=1000,
    color=ti.Vector([0.3, 0.3, 0.3]),
)

print("pushed bound parts: " + str(bound_part_num))

# /// --- END OF INIT OBJECT --- ///

search_template = tsph.Neighb_search_template(
    dim=config_space.dim[None],
    search_range=1,
)

fluid_neighb_grid = tsph.Neighb_grid( 
    #zxc fluid 流体/ bound 边界粒子  领域网格
    obj=fluid,
    dim=config_space.dim[None],
    lb=config_space.lb,
    rt=config_space.rt,
    cell_size=config_discre.part_size[None] * 2,
)

bound_neighb_grid = tsph.Neighb_grid(
    obj=bound,
    dim=config_space.dim[None],
    lb=config_space.lb,
    rt=config_space.rt,
    cell_size=config_discre.part_size[None] * 2,
)

fluid_neighb_grid.register(obj_pos=fluid.basic.pos)
bound_neighb_grid.register(obj_pos=bound.basic.pos)

# /// --- INIT SOLVER --- ///
# /// assign solver ///
fluid_df_solver = DFSPH(
    obj=fluid,
    dt=config_discre.dt[None],
    background_neighb_grid=fluid_neighb_grid,
    search_template=search_template,
    port_sph_psi="implicit_sph.sph_compression_ratio",  #zxc 密度，Psi
    port_rest_psi="implicit_sph.one",#zxc
    port_X="basic.rest_volume",  #zxc
)
bound_df_solver = DFSPH(
    obj=bound,
    dt=config_discre.dt[None],
    background_neighb_grid=bound_neighb_grid,
    search_template=search_template,
    port_sph_psi="implicit_sph.sph_compression_ratio",
    port_rest_psi="implicit_sph.one",
    port_X="basic.rest_volume",
)
#1创建fluid、grid；2 register grid；3 创建solver

# /// --- END OF INIT SOLVER --- ///

def loop():#zxc 逐时间步
    # /// dynamic dt ///
    tsph.cfl_dt(
        obj=fluid,
        obj_size=fluid.basic.size,
        obj_vel=fluid.basic.vel,
        cfl_factor=config_discre.cfl_factor,
        standard_dt=standart_dt,
        output_dt=config_discre.dt,
        output_inv_dt=config_discre.inv_dt,
    )
    #zxc CFL步长，dt=
    fluid_df_solver.update_dt(config_discre.dt[None])
    bound_df_solver.update_dt(config_discre.dt[None])

    # /// neighb search ///
    fluid_neighb_grid.register(obj_pos=fluid.basic.pos)
    bound_neighb_grid.register(obj_pos=bound.basic.pos)

      #  node_num 最大粒子数
      #  stack_top 实际粒子数



    #zxc from里面包含邻域搜索，并行化。
    # psi[0~stack top]=0
    # for 粒子  pid
    #     for neighborCell 
    #         for neighborPart
    #obj_sph_psi psi+=
    fluid_df_solver.clear_psi()
    ##clear,  deltavor=vor(n),deltavor=vor(n+1), deltavor-=-vor~

    ##
    fluid_df_solver.compute_psi_from(fluid_df_solver)
    fluid_df_solver.compute_psi_from(bound_df_solver)

    bound_df_solver.clear_psi()
    bound_df_solver.compute_psi_from(fluid_df_solver)
    bound_df_solver.compute_psi_from(bound_df_solver)

    #alpha=
    fluid_df_solver.clear_alpha()
    fluid_df_solver.compute_alpha_1_from(fluid_df_solver)#what
    fluid_df_solver.compute_alpha_1_from(bound_df_solver)
    fluid_df_solver.compute_alpha_2_from(fluid_df_solver)
    fluid_df_solver.compute_alpha_self()

    bound_df_solver.clear_alpha()
    bound_df_solver.compute_alpha_2_from(fluid_df_solver)
    bound_df_solver.compute_alpha_self()

    #参数 α 仅与当前位置分布相关，因此在迭代开始之前，可以预先计算

    
    #vel_adv=vel
    fluid_df_solver.set_vel_adv()
    #add 
    fluid_df_solver.clear_vor()
    fluid_df_solver.calc_vor(fluid_df_solver)

    #non pressure force
    fluid_df_solver.clear_acc()
    #acc=
    fluid_df_solver.add_acc(config_sim.gravity)
    #acc+=
    fluid_df_solver.add_acc_from_vis(       # viscosity
        kinetic_vis_coeff=config_sim.kinematic_vis,
        from_solver=fluid_df_solver,
    )
    #vel_adv+=acc*dt
    fluid_df_solver.update_vel_adv_from_acc()

    #zxc pressure Force,更新vel_adv
    fluid_df_solver.comp_iter_count[None] = 0
    while fluid_df_solver.is_compressible():
        fluid_df_solver.comp_iter_count[None] += 1


        #zxc dPsi=psi - rest psi
        fluid_df_solver.compute_delta_psi_self()
        bound_df_solver.compute_delta_psi_self()

        #zxc dPsi+=  
        #zxc Constant Density Solver step4: calc ρ*
        fluid_df_solver.compute_delta_psi_advection_from(fluid_df_solver)
        fluid_df_solver.compute_delta_psi_advection_from(bound_df_solver)
        bound_df_solver.compute_delta_psi_advection_from(fluid_df_solver)

        fluid_df_solver.ReLU_delta_psi()    #zxc psi正值
        bound_df_solver.ReLU_delta_psi()

        #zxc psi error过大则循环;更新vel_adv
        #zxc comp_avg_ratio=
        fluid_df_solver.check_if_compressible()
        bound_df_solver.check_if_compressible()

        #  delta density -> pressure force -> vel_adv 
        #vel_adv+=
        fluid_df_solver.update_vel_adv_from(fluid_df_solver)
        fluid_df_solver.update_vel_adv_from(bound_df_solver)


    #zxc vel = vel_adv
    fluid.attr_set_arr(
        obj_attr=fluid_df_solver.obj_vel,
        val_arr=fluid_df_solver.obj_vel_adv,
    )

    #zxc pos+=vel*dt
    fluid_df_solver.time_integral_arr(
        obj_frac=fluid_df_solver.obj_vel,
        obj_output_int=fluid_df_solver.obj_pos,
    )



loop()
""" GUI """
gui = tsph.Gui(config_gui)
gui.env_set_up()
loop_count = 0
while gui.window.running:
    if gui.op_system_run:
        loop()
        loop_count += 1
        print(loop_count)
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(fluid, size=config_discre.part_size[None])
        if gui.show_bound:
            gui.scene_add_parts(bound, size=config_discre.part_size[None])
        gui.scene_render()
