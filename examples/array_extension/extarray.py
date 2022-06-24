# A tutorial on writing a custom array implementation and reuse numba's numpy
# array implementation.
#
# This file is divided into multiple parts and introduces key ideas
# incrementally. There are tests that exercise the feature added in each part.
#
# Test with `pytest`.

import pytest

from typing import Tuple
import ctypes

from numba.core import types


import operator
from llvmlite import ir as llvmir
from llvmlite import binding as llvm
from numba import njit, typeof
from numba.extending import (
    register_model,
    models,
    typeof_impl,
    overload,
    overload_method,
    overload_classmethod,
    intrinsic,
    overload_attribute,
)
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core import cgutils
from numba.core import typing
import numpy as np
from numba.core.unsafe import refcount
from numba.np import numpy_support
from numba.core.pythonapi import _BoxContext, _UnboxContext
from numba.np.arrayobj import populate_array
from numba.core.errors import TypingError

import extarray_capi

# ----------------------------------------------------------------------------
# Part 1: Setup basic `ExtArray` class

# Define a python wrapper of the array object


class ExtArray:
    """This is the python wrapper of the array object defined in libextarray.so.
    This is a simple array implementation is always C-contiguous.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        handle: extarray_capi.ExtArrayHandlePtr,
    ):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.handle = handle
        self.size = np.prod(self.shape)
        self.nbytes = extarray_capi.getnbytes(handle)

    def __del__(self):
        extarray_capi.free(self.handle)

    @property
    def ndim(self):
        """number of dimension"""
        return len(self.shape)

    @property
    def layout(self):
        # Always C layout
        return "C"

    @property
    def handle_addr(self) -> int:
        """Returns the handle address
        """
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    @property
    def data_addr(self) -> int:
        """Returns the data address of the handle
        """
        return extarray_capi.getpointer(self.handle)

    def as_numpy(self):
        """Returns a numpy.ndarray wrapping the buffer of this array.
        """
        buf = (ctypes.c_byte * self.nbytes).from_address(
            extarray_capi.getpointer(self.handle)
        )
        return np.ndarray(shape=self.shape, dtype=self.dtype, buffer=buf)

    @property
    def strides(self):
        return self.as_numpy().strides

    @property
    def itemsize(self):
        return self.dtype.itemsize

    def __eq__(self, other):
        return all(
            [
                self.shape == other.shape,
                self.dtype == other.dtype,
                self.handle_addr == other.handle_addr,
            ]
        )

    def __repr__(self):
        fields = ", ".join(
            [
                f"shape={self.shape}",
                f"data=0x{self.data_addr:x}",
                f"handle=0x{self.handle_addr:x}",
                f"refct={self.refcount}",
            ]
        )
        return f"ExtArray({fields})"

    @property
    def refcount(self):
        return extarray_capi.getrefcount(self.handle)


def test_extarray_basic():
    """Test ExtArray class
    """
    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)
    assert ea.handle == handle

    ptr = extarray_capi.getpointer(ea.handle)
    assert ea.handle != ptr
    print("pointer", hex(ptr))

    cbuf = ea.as_numpy()
    # assert all the memory are zero'ed
    print(ea)
    print(cbuf)
    np.testing.assert_array_equal(cbuf, 0)


# ----------------------------------------------------------------------------
# Part 2: Setup Numba type for ExtArray and numba.typeof


class ExtArrayType(types.Array):
    """A Numba type of ExtArray"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"ExtArrayType({self.name})"
        if self.layout != "C":
            raise ValueError("ExtArrayType can only be of C contiguous")

    def as_base_array_type(self):
        return types.Array(dtype=self.dtype, ndim=self.ndim, layout=self.layout)

    # Needed for overloading operators and ufuncs
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if ufunc == np.add:
                if all(isinstance(inp, ExtArrayType) for inp in inputs):
                    return ExtArrayType
            else:
                # disable default numpy-based implementation
                return NotImplemented

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return type(self)(
            dtype=dtype,
            ndim=ndim,
            layout=layout,
            readonly=readonly,
            aligned=self.aligned,
        )


# Implement `typeof` for `ExtArray`.
# This affects `numba.typeof(x)`, which is used at function dispatching.
@typeof_impl.register(ExtArray)
def typeof_index(val, c):
    return ExtArrayType(typeof(val.dtype), val.ndim, val.layout)


def test_extarraytype_basic():
    """ExtArrayType and typeof
    """
    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)

    ea_typ = typeof(ea)
    print(ea_typ)

    assert ea_typ.layout == "C"
    assert ea_typ.ndim == 1
    assert ea_typ.dtype == typeof(np.dtype(np.float64))


# ----------------------------------------------------------------------------
# Part 3: Datamodel, box and unbox


# Define the datamodel for `ExtArrayType`.
# The base array data model is defined in `numba/core/datamodel/models.py`.


@register_model(ExtArrayType)
class _ExtArrayTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            # copied from base array type
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("parent", types.pyobject),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            ("data", types.CPointer(fe_type.dtype)),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
            # extra fields for ExtArray
            # NOTE: limitation of the _allocate API means this field will not be
            #       populated when the extarray is allocated by reused numpy API
            ("handle", types.voidptr),
        ]
        super().__init__(dmm, fe_type, members)


# Bind libextarray.so function to LLVM.
# Call these once to bind the library function to LLVM
llvm.add_symbol(
    "extarray_make_meminfo",
    ctypes.cast(extarray_capi.make_meminfo, ctypes.c_void_p).value,
)

llvm.add_symbol(
    "extarray_acquire",
    ctypes.cast(extarray_capi.acquire, ctypes.c_void_p).value,
)

llvm.add_symbol(
    "extarray_alloc", ctypes.cast(extarray_capi.alloc, ctypes.c_void_p).value
)

llvm.add_symbol(
    "extarray_getpointer",
    ctypes.cast(extarray_capi.getpointer, ctypes.c_void_p).value,
)

llvm.add_symbol(
    "extarray_meminfo_gethandle",
    ctypes.cast(extarray_capi.meminfo_gethandle, ctypes.c_void_p).value,
)


# Define a unboxer for `ExtArrayType`.
# The unboxer converts a Python object of `ExtArray` into the low-level
# representation specified by `_ExtArrayTypeModel`.


@unbox(ExtArrayType)
def unbox_extarray(typ, obj, c: _UnboxContext):
    # Get the ExtArrayHandle*
    handle_obj = c.pyapi.object_getattr_string(obj, "handle_addr")
    handle = c.builder.inttoptr(
        c.unbox(types.uintp, handle_obj).value, cgutils.voidptr_t
    )

    # Wrap the handle as a MemInfo
    meminfo_fnty = llvmir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t])
    meminfo_fn = cgutils.get_or_insert_function(
        c.builder.module, meminfo_fnty, name="extarray_make_meminfo"
    )
    meminfo = c.builder.call(meminfo_fn, [handle])

    # The MemInfo acquires a reference to the ExtArrayHandle
    acquire_fnty = llvmir.FunctionType(llvmir.VoidType(), [cgutils.voidptr_t])
    acquire_fn = cgutils.get_or_insert_function(
        c.builder.module, acquire_fnty, name="extarray_acquire"
    )
    c.builder.call(acquire_fn, [handle])

    # Extract fields from the ExtArray
    itemsize_obj = c.pyapi.object_getattr_string(obj, "itemsize")
    shape_obj = c.pyapi.object_getattr_string(obj, "shape")
    strides_obj = c.pyapi.object_getattr_string(obj, "strides")
    data_obj = c.pyapi.object_getattr_string(obj, "data_addr")

    # Populate the ExtArray low-level struct
    extarraycls = c.context.make_array(typ)
    extarray = extarraycls(c.context, c.builder)

    nd = typ.ndim

    populate_array(
        extarray,
        data=c.builder.inttoptr(
            c.unbox(types.uintp, data_obj).value, extarray.data.type
        ),
        shape=c.unbox(types.UniTuple(types.intp, nd), shape_obj).value,
        strides=c.unbox(types.UniTuple(types.intp, nd), strides_obj).value,
        itemsize=c.unbox(types.intp, itemsize_obj).value,
        meminfo=meminfo,
    )

    # The extra handle field.
    extarray.handle = handle

    aryptr = extarray._getpointer()

    def cleanup():
        c.pyapi.decref(itemsize_obj)
        c.pyapi.decref(shape_obj)
        c.pyapi.decref(strides_obj)
        c.pyapi.decref(data_obj)
        c.pyapi.decref(handle_obj)

    return NativeValue(
        c.builder.load(aryptr), is_error=c.pyapi.c_api_error(), cleanup=cleanup,
    )


# Define a boxer for `ExtArrayType`
# The boxer converts the low-level representation of `ExtArray` back into a
# `ExtArray` Python object.


def _box_extarray(shape, dtype, handle):
    hldr = ctypes.cast(ctypes.c_void_p(handle), extarray_capi.ExtArrayHandlePtr)
    return ExtArray(
        shape=shape, dtype=numpy_support.as_dtype(dtype), handle=hldr
    )


@box(ExtArrayType)
def box_extarray(typ, val, c: _BoxContext):
    # Setup accessor to the low-level extarray struct
    extarraycls = c.context.make_array(typ)
    extarray = extarraycls(c.context, c.builder, value=val)

    # Extract the handle from the meminfo
    gethandle_fnty = llvmir.FunctionType(
        cgutils.voidptr_t,
        [cgutils.voidptr_t, cgutils.voidptr_t, cgutils.intp_t],
    )
    gethandle_fn = cgutils.get_or_insert_function(
        c.builder.module, gethandle_fnty, "extarray_meminfo_gethandle",
    )
    handle = c.builder.call(
        gethandle_fn,
        [
            extarray.meminfo,
            c.builder.bitcast(extarray.data, cgutils.voidptr_t),
            c.builder.mul(extarray.itemsize, extarray.nitems),
        ],
    )

    lluintp = c.context.get_value_type(types.uintp)
    handle_obj = c.pyapi.long_from_longlong(c.builder.ptrtoint(handle, lluintp))

    # Prepare the array shape as a python tuple[int]
    shape_pyobjs = [
        c.box(types.intp, elem)
        for elem in cgutils.unpack_tuple(c.builder, extarray.shape)
    ]
    shape = c.pyapi.tuple_pack(shape_pyobjs)
    # Prepare the dtype as a python object
    dtype = c.pyapi.unserialize(c.pyapi.serialize_object(typ.dtype))
    # Setup call to _box_extarray in the Python interpreter.
    boxer = c.pyapi.unserialize(c.pyapi.serialize_object(_box_extarray))
    # Call _box_extarray
    retval = c.pyapi.call_function_objargs(boxer, [shape, dtype, handle_obj])

    # Cleanups
    for obj in shape_pyobjs:
        c.pyapi.decref(obj)
    c.pyapi.decref(shape)
    c.pyapi.decref(dtype)
    c.pyapi.decref(handle_obj)
    return retval


def test_unbox():
    @njit
    def foo(ea):
        # Unbox and check the reference count on the object
        return refcount.get_refcount(ea)

    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)

    refct = foo(ea)
    # There should be exactly one reference.
    assert refct == 1


def test_unbox_box():
    # Test with an identity function
    @njit
    def foo(ea):
        return ea

    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)

    ret = foo(ea)
    assert ea == ret


# ----------------------------------------------------------------------------
# Part 4: ExtArray allocator


## Expose extarray_empty to the JIT


# Define an equivalent to `np.empty` for the Python interpreter.
def extarray_empty(shape, dtype):
    nelem = np.prod(shape)
    dtype = np.dtype(dtype)
    handle = extarray_capi.alloc(nelem * dtype.itemsize)
    return ExtArray(shape=shape, dtype=dtype, handle=handle)


@intrinsic(prefer_literal=True)
def ext_array_alloc(typingctx, nbytes, nitems, ndim, shape, dtype, itemsize):
    if not isinstance(ndim, types.IntegerLiteral):
        # reject if ndim is not a literal
        return
    # note: skipping error checking for other arguments

    def codegen(context, builder, signature, args):
        [nbytes, nitems, ndim, shape, dtype, itemsize] = args

        # Call extarray_alloc to allocate a ExtArray handle
        alloc_fnty = llvmir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
        alloc_fn = cgutils.get_or_insert_function(
            builder.module, alloc_fnty, "extarray_alloc",
        )
        handle = builder.call(alloc_fn, [nbytes])

        # Call extarray_getpointer
        getptr_fnty = llvmir.FunctionType(
            cgutils.voidptr_t, [cgutils.voidptr_t]
        )
        getptr_fn = cgutils.get_or_insert_function(
            builder.module, getptr_fnty, "extarray_getpointer",
        )
        dataptr = builder.call(getptr_fn, [handle])

        extarraycls = context.make_array(signature.return_type)
        extarray = extarraycls(context, builder)

        # Make Meminfo
        meminfo_fnty = llvmir.FunctionType(
            cgutils.voidptr_t, [cgutils.voidptr_t]
        )
        meminfo_fn = cgutils.get_or_insert_function(
            builder.module, meminfo_fnty, name="extarray_make_meminfo"
        )
        meminfo = builder.call(meminfo_fn, [handle])

        # compute strides
        strides = []
        cur_stride = itemsize
        for s in reversed(cgutils.unpack_tuple(builder, shape)):
            strides.append(cur_stride)
            cur_stride = builder.mul(cur_stride, s)
        strides.reverse()

        populate_array(
            extarray,
            data=builder.bitcast(dataptr, extarray.data.type),
            shape=shape,
            strides=strides,
            itemsize=itemsize,
            meminfo=meminfo,
        )
        return extarray._getvalue()

    arraytype = ExtArrayType(
        ndim=ndim.literal_value, dtype=dtype.dtype, layout="C"
    )
    sig = typing.signature(
        arraytype, nbytes, nitems, ndim, shape, dtype, itemsize
    )
    return sig, codegen


@overload(extarray_empty)
def ol_empty_impl(shape, dtype):
    dtype = numpy_support.as_dtype(dtype.dtype)
    itemsize = dtype.itemsize

    def impl(shape, dtype):
        nelem = np.prod(np.asarray(shape))
        return ext_array_alloc(
            nelem * itemsize, nelem, len(shape), shape, dtype, itemsize
        )

    return impl


def test_allocator():
    @njit
    def foo(shape):
        return extarray_empty(shape, dtype=np.float64)

    shape = (2, 3)
    r = foo(shape)
    arr = r.as_numpy()
    assert arr.shape == shape
    assert arr.dtype == np.dtype(np.float64)
    assert arr.size == np.prod(shape)


# ----------------------------------------------------------------------------
# Part 4: Getitem Setitem


@overload_method(ExtArrayType, "as_numpy")
def extarray_as_numpy(arr):
    def impl(arr):
        return intrin_otherarray_as_numpy(arr)

    return impl


@intrinsic
def intrin_otherarray_as_numpy(typingctx, arr):
    base_arry_t = arr.as_base_array_type()

    def codegen(context, builder, signature, args):
        [arr] = args
        arry_t = signature.args[0]
        nativearycls = context.make_array(arry_t)
        nativeary = nativearycls(context, builder, value=arr)

        base_ary = context.make_array(base_arry_t)(context, builder)
        cgutils.copy_struct(base_ary, nativeary)
        out = base_ary._getvalue()
        context.nrt.incref(builder, base_arry_t, out)
        return out

    sig = typing.signature(base_arry_t, arr)
    return sig, codegen


def test_setitem():
    @njit
    def foo(size):
        arr = extarray_empty((size,), dtype=np.float64)
        for i in range(arr.size):
            arr[i] = i
        return arr

    r = foo(10)
    arr = r.as_numpy()
    np.testing.assert_equal(arr, np.arange(10))


def test_getitem():
    @njit
    def foo(size):
        arr = extarray_empty((size,), dtype=np.float64)
        for i in range(arr.size):
            arr[i] = i
        c = 0
        for i in range(arr.size):
            c += i
        return c

    res = foo(10)
    assert res == np.arange(10).sum()


def test_getitem_slice():
    @njit
    def foo(arr):
        return arr[1]

    arr = extarray_empty((2, 10), dtype=np.float64)
    arr.as_numpy()[:] = raw = np.arange(20).reshape(2, 10)
    res = foo(arr)
    np.testing.assert_equal(res.as_numpy(), raw[1])


def test_getitem_slice_unsupported_layout():
    @njit
    def foo(arr):
        return arr[:, 1]

    arr = extarray_empty((2, 10), dtype=np.float64)
    arr.as_numpy()[:] = np.arange(20).reshape(2, 10)
    with pytest.raises(TypingError) as e:
        # Slicing to non-C contiguous is not supported
        foo(arr)
    e.match("ExtArrayType can only be of C contiguous")


# ----------------------------------------------------------------------------
# Part 5: Access to handle address in ExtArray


@intrinsic
def intrin_extarray_handle_addr(typingctx, arr):
    def codegen(context, builder, signature, args):
        [arr] = args
        nativearycls = context.make_array(signature.args[0])
        nativeary = nativearycls(context, builder, value=arr)
        return nativeary.handle

    sig = typing.signature(types.voidptr, arr)
    return sig, codegen


@overload_attribute(ExtArrayType, "handle_addr")
def extarray_handle_addr(arr):
    def get(arr):
        return intrin_extarray_handle_addr(arr)

    return get


def test_handle_addr():
    @njit
    def foo(arr):
        return arr.handle_addr

    arr = extarray_empty((10,), dtype=np.float64)
    handle_addr = foo(arr)
    assert arr.handle_addr == handle_addr


# ----------------------------------------------------------------------------
# Part 6: Implement extarray add by reusing ndarray + ndarray

# Internally, Numba's numpy ndarray implementation expects the array type
# to have a `_allocate()` classmethod. Defining this will enable reuse of
# existing numpy ndarray code.

# See `ExtArrayType.__array_ufunc__`. It enables ExtArrayType to reuse the
# internal ndarray add inside Numba.


def extarray_new_meminfo(builder, nbytes):
    # Call extarray_alloc to allocate a ExtArray handle
    alloc_fnty = llvmir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
    alloc_fn = cgutils.get_or_insert_function(
        builder.module, alloc_fnty, "extarray_alloc",
    )
    handle = builder.call(alloc_fn, [nbytes])

    # Make Meminfo
    meminfo_fnty = llvmir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t])
    meminfo_fn = cgutils.get_or_insert_function(
        builder.module, meminfo_fnty, name="extarray_make_meminfo"
    )
    meminfo = builder.call(meminfo_fn, [handle])
    return meminfo


@intrinsic
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array
    """

    def codegen(context, builder, signature, args):
        [allocsize, align] = args
        # Note: align is being ignored for now
        meminfo = extarray_new_meminfo(builder, allocsize)
        return meminfo

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = typing.signature(mip, allocsize, align)
    return sig, codegen


@overload_classmethod(ExtArrayType, "_allocate")
def oat_allocate(cls, allocsize, alignment):
    def impl(cls, allocsize, alignment):
        return intrin_alloc(allocsize, alignment)

    return impl


@njit
def extarray_arange(size, dtype):
    out = extarray_empty((size,), dtype=np.float64)
    for i in range(size):
        out[i] = i
    return out


def test_add():
    @njit
    def foo(n):
        a = extarray_arange(n, dtype=np.float64)
        b = extarray_arange(n, dtype=np.float64)
        res = a + b
        return res

    n = 12
    res = foo(12)
    assert isinstance(res, ExtArray)
    np.testing.assert_equal(res.as_numpy(), np.arange(n) + np.arange(n))


# ----------------------------------------------------------------------------
# Part 7: Implement extarray subtract with a custom overload version

# See ExtArrayType.__array_ufunc__. It disables the reuse of the internal
# subtract implementation in Numba.

# Define an overload for subtraction


@overload(operator.sub)
def ol_sub_impl(lhs, rhs):
    """Implement subtract differently.

    This will compute `2 * lhs - 2 * rhs`.
    """
    if isinstance(lhs, ExtArrayType):
        if lhs.dtype != rhs.dtype:
            raise TypeError(
                f"LHS dtype ({lhs.dtype}) != RHS.dtype ({rhs.dtype})"
            )

        def impl(lhs, rhs):
            if lhs.shape != rhs.shape:
                raise ValueError("shape incompatible")
            out = extarray_empty(lhs.shape, lhs.dtype)
            for i in np.ndindex(lhs.shape):
                # Do a different thing so we know this version is used.
                out[i] = 2 * lhs[i] - 2 * rhs[i]
            return out

        return impl


def test_sub():
    @njit
    def foo(n):
        a = extarray_arange(n, dtype=np.float64)
        b = extarray_arange(n, dtype=np.float64)
        res = a - b
        return res

    n = 12
    res = foo(12)
    assert isinstance(res, ExtArray)
    np.testing.assert_equal(res.as_numpy(), 2 * np.arange(n) - 2 * np.arange(n))
