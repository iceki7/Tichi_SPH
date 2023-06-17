import taichi as ti
from ti_sph import *
from part_template import part_template
from grid_template import grid_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda, device_memory_GB=3) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

''' GLOBAL SETTINGS '''
part_size = 0.008
max_time_step = part_size/20
world = World(dim=2)
world.set_part_size(part_size)
world.set_dt(max_time_step)

'''BASIC SETTINGS FOR FLUID'''
fluid_rest_density = val_f(1000)
cube_data = Cube_data(lb=vec2f(-2, -4+part_size), rt=vec2f(-0.2, 0), span=world.g_part_size[None]*1.001)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_num = val_i(cube_data.num)
fluid_part_1 = world.add_part_obj(part_num=fluid_part_num[None], is_dynamic=True)
fluid_part_1.instantiate_from_template(part_template)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_1.open_stack(val_i(cube_data.num))
fluid_part_1.fill_open_stack_with_nparr(fluid_part_1.pos, cube_data.pos)
fluid_part_1.fill_open_stack_with_val(fluid_part_1.size, fluid_part_1.get_part_size())
fluid_part_1.fill_open_stack_with_val(fluid_part_1.volume, val_f(fluid_part_1.get_part_size()[None]**world.g_dim[None]))
fluid_part_1.fill_open_stack_with_val(fluid_part_1.mass, val_f(fluid_rest_density[None]*fluid_part_1.get_part_size()[None]**world.g_dim[None]))
fluid_part_1.fill_open_stack_with_val(fluid_part_1.rest_density, fluid_rest_density)
fluid_part_1.close_stack()

'''INIT AN FLUID PARTICLE OBJECT'''
cube_data.translate(vec2f(2.2, 0))
fluid_part_num = val_i(cube_data.num)
fluid_part_2 = world.add_part_obj(part_num=fluid_part_num[None], is_dynamic=True)
fluid_part_2.instantiate_from_template(part_template)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_2.open_stack(val_i(cube_data.num))
fluid_part_2.fill_open_stack_with_nparr(fluid_part_2.pos, cube_data.pos)
fluid_part_2.fill_open_stack_with_val(fluid_part_2.size, fluid_part_2.get_part_size())
fluid_part_2.fill_open_stack_with_val(fluid_part_2.volume, val_f(fluid_part_2.get_part_size()[None]**world.g_dim[None]))
fluid_part_2.fill_open_stack_with_val(fluid_part_2.mass, val_f(fluid_rest_density[None]*fluid_part_2.get_part_size()[None]**world.g_dim[None]))
fluid_part_2.fill_open_stack_with_val(fluid_part_2.rest_density, fluid_rest_density)
fluid_part_2.close_stack()

''' INIT BOUNDARY PARTICLE OBJECT '''
box_data = Box_data(lb=vec2f(-4, -4), rt=vec2f(4, 4), span=world.g_part_size[None]*1.05, layers=3)
bound_rest_density = val_f(1000)
bound_part = world.add_part_obj(part_num=box_data.num, is_dynamic=False)
bound_part.instantiate_from_template(part_template)

bound_part.open_stack(val_i(box_data.num))
bound_part.fill_open_stack_with_arr(bound_part.pos, box_data.pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.get_part_size())
bound_part.fill_open_stack_with_val(bound_part.volume, val_f(bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.mass, val_f(bound_rest_density[None]*bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.rest_density, bound_rest_density)
bound_part.close_stack()

'''INIT NEIGHBOR SEARCH OBJECTS'''
fluid_part_1.add_module_neighb_search()
fluid_part_2.add_module_neighb_search()
bound_part.add_module_neighb_search()

fluid_part_1.add_neighb_objs(world.part_obj_list)
fluid_part_2.add_neighb_objs(world.part_obj_list)
bound_part.add_neighb_objs(world.part_obj_list)

fluid_part_1.add_solver_adv()
fluid_part_2.add_solver_adv()

fluid_part_1.add_solver_sph()
fluid_part_2.add_solver_sph()
bound_part.add_solver_sph()

fluid_part_1.add_solver_df(div_free_threshold=3e-4)
fluid_part_2.add_solver_df(div_free_threshold=3e-4)
bound_part.add_solver_df(div_free_threshold=3e-4)

world.init_modules()

world.neighb_search()

# sense_grid = Sense_grid(type=Sense_grid.FIXED_GRID, neighb_pool_size=val_i(3e6),world=world, cell_size=val_f(0.1))
sense_grid = Sense_grid(type=Sense_grid.FIXED_RES, neighb_pool_size=val_i(3e6), world=world, cell_size=val_f(0.1), grid_res=val_i(64), grid_center=vec2_f([0, 0]))
pos_np = sense_grid.node_index.to_numpy().astype(np.int32)
# save pos_np to txt file
np.savetxt("pos_np.txt", pos_np, fmt='%i')

# sense_grid.add_sensed_particles(fluid_part_1)
# sense_grid.add_sensed_particles(fluid_part_2)
sense_grid.step()
sense_output = Output_manager(format_type = Output_manager.type.GRID, data_source = sense_grid)
sense_output.add_output_dataType("pos",2)
sense_output.export_to_numpy()
print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop():
    world.update_pos_in_neighb_search()

    world.neighb_search()
    world.step_sph_compute_density()
    world.step_df_compute_alpha()
    world.step_df_div()
    print('div_free iter:', fluid_part_1.m_solver_df.div_free_iter[None])

    sense_grid.step()

    world.clear_acc()
    world.add_acc_gravity()
    world.acc2vel_adv()

    world.step_df_incomp()
    print('incomp iter:', fluid_part_1.m_solver_df.incompressible_iter[None])

    world.update_pos_from_vel()

    world.cfl_dt(0.5, max_time_step)

    print(' ')

def run(loop):
    gui = Gui3d()

    fps = 60
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while gui.window.running:

        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            # print('loop count', loop_count, 'compressible ratio', 'incompressible iter', fluid_part_1.m_solver_df.incompressible_iter[None], ' ', fluid_part_2.m_solver_df.incompressible_iter[None])
            # print('comp ratio', fluid_part_1.m_solver_df.compressible_ratio[None], ' ', fluid_part_2.m_solver_df.compressible_ratio[None])
            # print('dt', world.g_dt[None])
        
        if gui.op_refresh_window:
            gui.scene_setup()
            if gui.show_bound:
                gui.scene_add_parts(obj_pos=fluid_part_1.pos, obj_color=(1,0.5,0),index_count=fluid_part_1.get_stack_top()[None],size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=fluid_part_2.pos, obj_color=(0,0.5,1),index_count=fluid_part_2.get_stack_top()[None],size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
            else:
                gui.scene_add_parts_colorful(obj_pos=sense_grid.pos, obj_color=sense_grid.clampped_rgb, index_count=sense_grid.get_stack_top()[None], size=sense_grid.get_part_size()[None]*0.5)
            
            gui.canvas.scene(gui.scene)  # Render the scene

            if(sim_time > timer*inv_fps):
                if gui.op_save_img:
                    gui.window.save_image('output/'+str(timer)+'.png')
                timer += 1

            gui.window.show()
    
        if timer > 660:
            break

loop()
run(loop)







