import taichi as ti
from ..basic_op.type import *

@ti.data_oriented
class Particle:
    def __init__(
        self,
        part_num: int,
        Dynamic: bool = True,
    ):
        if (part_num <= 0):
            raise ValueError("part_num must be larger than 0")

        self.id = val_i(-1)

        self.part_num_ = val_i(part_num)
        self.stack_top_ = val_i(0)
        
        self.is_dynamic = True

        self.attr_list = {}
        self.array_list = {}
        self.struct_list = {}

        # seq_log to track the order of particles (and the swap of particles)
        self.seq_log_ = ti.field(ti.i32, self.part_num_[None])
        for i in range(self.part_num_[None]):
            self.seq_log_[i] = i

    # Functions Type 1: structure management
    def add_attr(self, name, attr):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if (isinstance(attr, list)):
                self.__dict__[name] = []
                self.attr_list[name] = []
                for attr_i in attr:
                    self.__dict__[name].append(attr_i)

            else:
                self.__dict__[name] = attr

            self.attr_list[name] = self.__dict__[name]

    # This function is deprecated (has been merged with add_attr()) and will be removed in the future.
    def add_attrs(self, name, attrs):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            self.__dict__[name] = []
            self.attr_list[name] = []
            for attr in attrs:
                self.__dict__[name].append(attr)
            self.attr_list[name] = self.__dict__[name]

    def add_array(self, name, array, bundle=1):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if (isinstance(array, list)):
                self.__dict__[name] = []
                self.array_list[name] = []
                if (bundle == 1):
                    for array_i in array:
                        ti.root.dense(ti.i, self.part_num_[None]).place(array_i)
                        self.__dict__[name].append(array_i)
                else:
                    for array_i in array:
                        ti.root.dense(ti.ij, (self.part_num_[None], bundle)).place(array_i)
                        self.__dict__[name].append(array_i)
            else:
                if (bundle == 1):
                    ti.root.dense(ti.i, self.part_num_[None]).place(array)
                else:
                    ti.root.dense(ti.ij, (self.part_num_[None], bundle)).place(array)
                self.__dict__[name] = array

            self.array_list[name] = self.__dict__[name]

    # This function is deprecated (has been merged with add_array()) and will be removed in the future.
    def add_arrays(self, name, arrays):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            self.__dict__[name] = []
            self.array_list[name] = []
            for array in arrays:
                ti.root.dense(ti.i, self.part_num_[None]).place(array)
                self.__dict__[name].append(array)
            self.array_list[name] = self.__dict__[name]

    def add_struct(self, name, struct, bundle=1):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if (isinstance(struct, list)):
                self.__dict__[name] = []
                self.struct_list[name] = []
                if (bundle == 1):
                    for struct_i in struct:
                        self.__dict__[name].append(struct_i.field(shape=(self.part_num_[None],)))
                else:
                    for struct_i in struct:
                        self.__dict__[name].append(struct_i.field(shape=(self.part_num_[None], bundle)))
            else:
                if (bundle == 1):
                    self.__dict__[name] = struct.field(shape=(self.part_num_[None],))
                else:
                    self.__dict__[name] = struct.field(shape=(self.part_num_[None], bundle))
            self.struct_list[name] = self.__dict__[name]

    def add_structs(self, name, structs):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")

        self.__dict__[name] = []
        self.struct_list[name] = []
        for struct in structs:
            self.__dict__[name].append(struct.field(shape=(self.part_num_[None],)))
        self.struct_list[name] = self.__dict__[name]

    # Functions Type 2: verbose functions

    def verbose_structs(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template = "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} structs: ")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.struct_list:
                if (isinstance(self.__dict__[name], list)):
                    iter = 0
                    for struct in self.__dict__[name]:
                        print(head_template.format(f"{name}[{iter}] {struct.shape}"))
                        for key in struct.keys:
                            # print('self.' + name + '[' + str(iter) + ']' + '.' + key  + '.dtype')
                            print(sub_element_template.format(
                                key,
                                f"dtype={eval('self.' + name + '[' + str(iter) + ']' + '.' + key  + '.dtype')}",
                                f"{[eval('self.' + name + '[' + str(iter) + ']' + '.' + key)]}"))
                        iter += 1
                        print(dash_template.format())
                else:
                    print(head_template.format(f"{name}{self.__dict__[name].shape}"))
                    for key in self.__dict__[name].keys:
                        print(sub_element_template.format(
                            key, f"dtype={eval('self.'+name+'.'+key).dtype}", f"{[eval('self.'+name+'.'+key)]}"))
                    print(dash_template.format())

    def verbose_arrays(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        element_template = "{0:16}\t|\t{1:8}\t|\t{2:16}"
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template = "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} arrays:")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.array_list:
                if (isinstance(self.__dict__[name], list)):
                    print(head_template.format(f"{name}"))
                    iter = 0
                    for array in self.__dict__[name]:
                        print(sub_element_template.format(
                            f"[{iter}] {array.shape}", f"dtype={array.dtype}", f"{[array]}"))
                        iter += 1
                else:
                    print(element_template.format(
                        f"{name} {self.__dict__[name].shape}", f"dtype={self.__dict__[name].dtype}", f"{[self.__dict__[name]]}"))
                print(dash_template.format())
        print(" ")

    def verbose_attrs(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        element_template = "{0:16}\t|\t{1:8}\t|\t{2:16}"
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template = "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} attrs:")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.attr_list:
                if (isinstance(self.__dict__[name], list)):
                    print(head_template.format(name))
                    iter = 0
                    for attr in self.__dict__[name]:
                        print(sub_element_template.format(
                            f"[{iter}]", f"type={type(attr)}", f"{attr}"))
                        iter += 1
                else:
                    print(element_template.format(
                        name, f"type={type(self.__dict__[name])}", f"{self.__dict__[name]}"))
                print(dash_template.format())
        print(" ")

    # Functions Type 3: Data operations
    def update_stack_top(self, num: int):
        self.stack_top_[None] += num

    @ti.kernel
    def clear(self, attr_: ti.template()):
        for i in range (self.stack_top_[None]):
            attr_[i] *= 0

    def set_from_numpy(self, to: ti.template(), data: ti.types.ndarray()):
        num = data.shape[0]
        arr = to.to_numpy()
        arr[self.stack_top_[None]:num, :] = data
        to.from_numpy(arr)

    @ti.kernel
    def push_from_val_(self, to_: ti.template(), num: ti.i32, val_: ti.template()):
        for i in range(num):
            to_[i+self.stack_top_[None]] = val_[None]

    @ti.kernel
    def add_from_valf(self, to_: ti.template(), valf:ti.f32):
        for i in range(self.stack_top_[None]):
            to_[i] += valf
    
    @ti.kernel
    def set_from_valf(self, to_: ti.template(), valf:ti.f32):
        for i in range(self.stack_top_[None]):
            to_[i] = valf
    
    @ti.kernel
    def add_from_vali(self, to_: ti.template(), vali:ti.i32):
        for i in range(self.stack_top_[None]):
            to_[i] += vali
    
    @ti.kernel
    def set_from_vali(self, to_: ti.template(), vali:ti.i32):
        for i in range(self.stack_top_[None]):
            to_[i] = vali