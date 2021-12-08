# Numba extension for cuRAND functions. Presently only implements:
#
# - curand_init()
# - curand()
#
# This is enough for a proof-of-concept embedded calls to cuRAND functions in
# Numba kernels.

from numba import cuda, types
from numba.core.extending import (lower_builtin, models, register_model,
                                  typeof_impl)
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cuda.cudadecl import register_global

import numpy as np
import operator


# cuRAND state type as a NumPy dtype - this mirrors the state defined in
# curand_kernel.h. Can be used to inspect the state through the device array
# held by CurandStates.

state_fields = [
    ('d', np.int32),
    ('v', np.int32, 5),
    ('boxmuller_flag', np.int32),
    ('boxmuller_flag_double', np.int32),
    ('boxmuller_extra', np.float32),
    ('boxmuller_extra_double', np.float64),
]

curandState = np.dtype(state_fields, align=True)


# Hold an array of cuRAND states - somewhat analagous to a curandState* in
# C/C++.

class CurandStates:
    def __init__(self, n):
        self._array = cuda.device_array(n, dtype=curandState)

    @property
    def data(self):
        return self._array.__cuda_array_interface__['data'][0]


# Numba typing for cuRAND state. Generally we treat cuRAND states as a struct
# mirroring the C/C++ struct, and an array of states is a pointer with slightly
# special behaviour - doing a getitem on a state array returns a reference
# (pointer) to that element, because we need to pass pointers to individual
# elements to cuRAND functions.

class CurandState(types.Type):
    def __init__(self):
        super().__init__(name='CurandState')


curand_state = CurandState()


class CurandStatePointer(types.Type):
    def __init__(self):
        self.dtype = curand_state
        super().__init__(name='CurandState*')


curand_state_pointer = CurandStatePointer()


@typeof_impl.register(CurandStates)
def typeof_curand_states(val, c):
    return curand_state_pointer


# The CurandState model mirrors the C/C++ structure, and the state pointer
# represented similarly to other pointers.

@register_model(CurandState)
class curand_state_model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('d', types.int32),
            ('v', types.UniTuple(types.int32, 5)),
            ('boxmuller_flag', types.int32),
            ('boxmuller_flag_double', types.int32),
            ('boxmuller_extra', types.float32),
            ('boxmuller_extra_double', types.float64),
        ]
        super().__init__(dmm, fe_type, members)


register_model(CurandStatePointer)(models.PointerModel)


# Typing for cuRAND states:
#
# - getitem on a CurandStatePointer returns another CurandStatePointer.
# - setitem on a CurandStatePointer with a CurandState copies the CurandState
#   to the item referred to by the index.

@register_global(operator.getitem)
class GetItemCurandStatePointer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if (isinstance(ptr, CurandStatePointer) and
                isinstance(idx, types.Integer)):
            i = types.intp if idx.signed else types.uintp
            return signature(ptr, ptr, i)


@register_global(operator.setitem)
class SetItemCurandStatePointer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr, idx, val = args
        if (isinstance(ptr, CurandStatePointer) and
                isinstance(idx, types.Integer) and
                isinstance(val, CurandStatePointer)):
            i = types.intp if idx.signed else types.uintp
            return signature(types.none, ptr, i, val)


# Lowering for cuRAND states, following the rules outlined above.

@lower_builtin(operator.getitem, CurandStatePointer, types.Integer)
def getitem_curand_states(context, builder, sig, args):
    base_ptr, idx = args
    elem_ptr = builder.gep(base_ptr, [idx])
    return elem_ptr


@lower_builtin(operator.setitem, CurandStatePointer, types.Integer,
               CurandStatePointer)
def setitem_curand_states(context, builder, sig, args):
    base_ptr, idx, val = args
    elem_ptr = builder.gep(base_ptr, [idx])
    builder.store(builder.load(val), elem_ptr)


# Numba forward declarations of cuRAND functions. These call shim functions
# prepended with _numba, that simply forward arguments to the named cuRAND
# function.

curand_init_sig = types.void(
    types.uint64,
    types.uint64,
    types.uint64,
    curand_state_pointer
)

curand_init = cuda.declare_device('_numba_curand_init', curand_init_sig)
curand = cuda.declare_device('_numba_curand',
                             types.uint32(curand_state_pointer))


# Argument handling. When a CurandStatePointer is passed into a kernel, we
# really only need to pass the pointer to the data, not the whole underlying
# array structure. Our handler here transforms these arguments into a uint64
# holding the pointer.

class CurandStateArgHandler:
    def prepare_args(self, ty, val, **kwargs):
        if isinstance(val, CurandStates):
            assert ty == curand_state_pointer
            return types.uint64, val.data
        else:
            return ty, val


curand_state_arg_handler = CurandStateArgHandler()
