import taichi as ti
from ..basic_op.type import *

@ti.data_oriented
class World:
    def __init__(self, dim=3):
        ''' GLOBAL CONFIGURATION '''
        self.dim = val_i(dim)
        self.space_lb = vecx_f(self.dim[None])
        self.space_rt = vecx_f(self.dim[None])
        self.gravity = vecx_f(self.dim[None])
        self.space_lb.fill(-8)
        self.space_rt.fill(8)
        self.gravity[None][1] = -9.8
        self.dt = val_f(0.001)
        self.part_size = val_f(0.1)
        self.avg_neighb_part_num = val_i(32)
        self.obj_num = val_i(3)

        self.dependent_init()

    def dependent_init(self):
        self.space_size = vecx_f(self.dim[None])
        self.space_center = vecx_f(self.dim[None])
        self.space_size[None] = self.space_rt[None] - self.space_lb[None]
        self.space_center[None] = (self.space_rt[None] + self.space_lb[None]) / 2

        self.part_volume = val_f(self.part_size[None] ** self.dim[None])
        self.support_radius = val_f(self.part_size[None] * 2)

    def refresh(self):
        self.dependent_init()
    
    def set_part_size(self, size):
        self.part_size = val_f(size)
        self.refresh()
    
    def set_time_step(self, dt):
        self.dt = val_f(dt)