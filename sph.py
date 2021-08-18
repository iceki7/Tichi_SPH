from os import system
import numpy as np
import time

from taichi.lang.ops import atomic_min
from sph_obj import *


@ti.kernel
def SPH_neighbour_loop_template(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        obj.sph_psi[i] += nobj.rest_volume[neighb_pid]*W((obj.pos[i] - nobj.pos[neighb_pid]).norm())

@ti.kernel
def SPH_clean_value(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.W[i] = 0
        obj.sph_compression[i] = 0
        obj.sph_density[i] = 0
        obj.alpha_2[i] = 0
        for j in ti.static(range(dim)):
            obj.W_grad[i][j] = 0
            obj.acce[i][j] = 0
            obj.acce_adv[i][j] = 0
            obj.alpha_1[i][j] = 0
            obj.pressure_force[i][j] = 0

@ti.kernel
def cfl_condition(obj: ti.template()):
    dt[None] = init_part_size/cs
    for i in range(obj.part_num[None]):
        v_norm = obj.vel[i].norm()
        if v_norm > 1e-4:
            atomic_min(dt[None], part_size[1]/obj.vel[i].norm())

@ti.kernel
def SPH_prepare_attr(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        Wr = W((obj.pos[i] - nobj.pos[neighb_pid]).norm())
                        obj.W[i] += Wr
                        obj.sph_compression[i] += Wr*nobj.rest_volume[neighb_pid]
                        obj.sph_density[i] += Wr*nobj.mass[neighb_pid]

@ti.kernel
def SPH_prepare_alpha_1(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.alpha_1[i] += nobj.X[neighb_pid] * xij/r*W_grad(r)

@ti.kernel
def SPH_prepare_alpha_2(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        r = (obj.pos[i] - nobj.pos[neighb_pid]).norm()
                        if r>0:
                            obj.alpha_2[i] += W_grad(r)**2 * nobj.X[neighb_pid]**2 / nobj.mass[neighb_pid]

@ti.kernel
def SPH_prepare_alpha(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.alpha[i] = obj.alpha_1[i].dot(obj.alpha_1[i])/obj.mass[i] + obj.alpha_2[i]
        if obj.alpha[i]<1e-4:
            obj.alpha[i]=1e-4

@ti.kernel
def SPH_advection_gravity_acc(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] += gravity[None]

@ti.kernel
def SPH_advection_viscosity_acc(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.acce_adv[i] += W_lap(xij, r, nobj.X[neighb_pid]/nobj.sph_psi[neighb_pid], obj.vel[i] - nobj.vel[neighb_pid])*dynamic_viscosity/obj.rest_density[i]

@ti.kernel
def WC_pressure_val(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pressure[i] = (obj.rest_density[i] * cs**2 / wc_gamma)*((obj.sph_density[i]/obj.rest_density[i])**7-1)
        if obj.pressure[i]<0:
            obj.pressure[i]=0
@ti.kernel
def WC_pressure_acce(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        p_term = obj.pressure[i]/((obj.sph_density[i])**2) + nobj.pressure[neighb_pid]/((nobj.sph_density[neighb_pid])**2)
                        if r>0:
                            obj.acce_adv[i] += -p_term * nobj.mass[neighb_pid] * xij/r * W_grad(r)

@ti.kernel
def IPPE_adv_psi_init(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.psi_adv[i] = obj.sph_psi[i] - obj.rest_psi[i]

@ti.kernel
def IPPE_adv_psi(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.psi_adv[i] += (xij/r*W_grad(r)).dot(obj.vel_adv[i] - nobj.vel_adv[neighb_pid]) * nobj.X[neighb_pid] * dt[None]

@ti.kernel
def IPPE_psi_adv_non_negative(obj: ti.template()):
    obj.compression[None] = 0
    for i in range(obj.part_num[None]):
        if obj.psi_adv[i]<0:
            obj.psi_adv[i] = 0
        obj.compression[None] += (obj.psi_adv[i] / obj.rest_psi[i])
    obj.compression[None] /= obj.part_num[None]
    
@ti.kernel
def IPPE_update_vel_adv(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.vel_adv[i] += -(1/dt[None]) * ((obj.psi_adv[i]*nobj.X[neighb_pid]/obj.alpha[i])+(
                                nobj.psi_adv[neighb_pid]*obj.X[i]/nobj.alpha[neighb_pid])) * (xij/r*W_grad(r)) / obj.mass[i]

@ti.kernel
def SPH_advection_update_vel_adv(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] = obj.vel[i] + obj.acce_adv[i]*dt[None]

@ti.kernel
def SPH_vel_2_vel_adv(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] = obj.vel[i]

@ti.kernel
def SPH_vel_adv_2_vel(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel[i] = obj.vel_adv[i]

@ti.kernel
def SPH_update_pos(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel[i] = obj.vel_adv[i]
        obj.pos[i] += obj.vel[i]*dt[None]

############################### main ###############################

""" init data structure """
ngrid = Ngrid()
fluid = Fluid(max_part_num=fluid_part_num)
bound = Fluid(max_part_num=bound_part_num)
for obj in obj_list:
    obj.set_zero()
    obj.ones.fill(1)

dt[None] = init_part_size/cs
phase_rest_density.from_numpy(np_phase_rest_density)
sim_space_lb.from_numpy(np_sim_space_lb)
sim_space_rt.from_numpy(np_sim_space_rt)
part_size.from_numpy(np_part_size)
sph_h.from_numpy(np_sph_h)
sph_sig.from_numpy(np_sph_sig)
gravity.from_numpy(np_gravity)
node_dim.from_numpy(np_node_dim)
node_dim_coder.from_numpy(np_node_dim_coder)
for i in range(np_neighb_template.shape[1]):
    for j in range(dim):
        neighb_template[i][j] = np_neighb_template[j][i]

""" setup scene """
lb = np.zeros(dim, np.float32)
rt = np.zeros(dim, np.float32)
mask = np.ones(dim, np.int32)
volume_frac = np.zeros(phase_num, np.float32)
""" push cube """
fluid.push_2d_cube(center_pos=[-1, 0], size=[1.8, 3.6], volume_frac=[1, 0], color=0x068587)
fluid.push_2d_cube([1,0],[1.8, 3.6],[0,1],0x8f0000)
bound.push_2d_cube([0,0],[5,5],[1,0],0xFF4500,4)

def sph_step():
    """ neighbour search """
    ngrid.clear_node()
    ngrid.encode(fluid)
    ngrid.encode(bound)
    ngrid.mem_shift()
    ngrid.fill_node(fluid)
    ngrid.fill_node(bound)
    """ SPH clean value """
    SPH_clean_value(fluid)
    SPH_clean_value(bound)
    """ SPH compute W and W_grad """
    SPH_prepare_attr(ngrid, fluid, fluid)
    SPH_prepare_attr(ngrid, fluid, bound)
    SPH_prepare_attr(ngrid, bound, bound)
    SPH_prepare_attr(ngrid, bound, fluid)
    SPH_prepare_alpha_1(ngrid, fluid, fluid)
    SPH_prepare_alpha_1(ngrid, fluid, bound)
    SPH_prepare_alpha_2(ngrid, fluid, fluid)
    SPH_prepare_alpha_2(ngrid, bound, fluid)
    SPH_prepare_alpha(fluid)
    SPH_prepare_alpha(bound)
    """ IPPE SPH divergence """
    SPH_vel_2_vel_adv(fluid)
    div_iter_count = 0
    while div_iter_count<iter_threshold_min or fluid.compression[None]>divergence_threshold:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        div_iter_count+=1
        if div_iter_count>iter_threshold_max:
            break
    SPH_vel_adv_2_vel(fluid)
    """ SPH advection """
    SPH_advection_gravity_acc(fluid)
    SPH_advection_viscosity_acc(ngrid, fluid, fluid)
    SPH_advection_update_vel_adv(fluid)
    """ IPPE SPH pressure """
    incom_iter_count = 0
    while incom_iter_count<iter_threshold_min or fluid.compression[None]>compression_threshold:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        incom_iter_count+=1
        if incom_iter_count>iter_threshold_max:
            break
    """ debug info """
    # print('iter div: ', div_iter_count)
    # print('incom div: ', incom_iter_count)
    """ WC SPH pressure """
    # WC_pressure_val(fluid)
    # WC_pressure_acce(ngrid, fluid, fluid)
    # WC_pressure_acce(ngrid, fluid, bound)
    # SPH_advection_update_vel_adv(fluid)
    """ SPH uodate pos """
    SPH_update_pos(fluid)
    """ SPH debug """

""" GUI system """
time_count = float(0)
time_counter = int(0)
print('fluid particle count: ', fluid.part_num[None])
print('bound particle count: ', bound.part_num[None])
gui = ti.GUI('SPH', to_gui_res(gui_res_0))
while gui.running and not gui.get_event(gui.ESCAPE):
    gui.clear(0x112F41)
    while time_count*refreshing_rate < time_counter:
        cfl_condition(fluid)
        sph_step()
        time_count += dt[None]
    time_counter += 1
    print('current time: ', time_count)
    print('time step: ', dt[None])
    gui.circles(to_gui_pos(fluid), radius=to_gui_radii(part_radii_relax), color=to_gui_color(fluid))
    gui.circles(to_gui_pos(bound), radius=to_gui_radii(part_radii_relax), color=to_gui_color(bound))
    gui.show(f"img\\{time_counter}_rf{refreshing_rate}.png")
