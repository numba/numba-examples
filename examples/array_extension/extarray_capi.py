import os
import ctypes

# Get libextarray.so relative to this file
_dirpath = os.path.dirname(__file__)
lib = ctypes.CDLL(os.path.join(_dirpath, 'libextarray.so'))


class ExtArrayHandle(ctypes.Structure):
    # opaque struct
    pass

ExtArrayHandlePtr = ctypes.POINTER(ExtArrayHandle)

# ExtArrayHandle* extarray_alloc(size_t nbytes)
lib.extarray_alloc.restype = ExtArrayHandlePtr
lib.extarray_alloc.argtypes = [ctypes.c_size_t]

# void extarray_free(ExtArrayHandle *hldr)
lib.extarray_free.restype = None
lib.extarray_free.argtypes = [ExtArrayHandlePtr]

# void* extarray_getpointer(ExtArrayHandle *hldr)
lib.extarray_getpointer.restype = ctypes.c_void_p
lib.extarray_getpointer.argtypes = [ExtArrayHandlePtr]

# size_t extarray_getrefcount(ExtArrayHandle *hldr)
lib.extarray_getrefcount.restype = ctypes.c_size_t
lib.extarray_getrefcount.argtypes = [ExtArrayHandlePtr]

# void* extarray_make_meminfo(ExtArrayHandle *hldr)
lib.extarray_make_meminfo.restype = ctypes.c_void_p
lib.extarray_make_meminfo.argtypes = [ExtArrayHandlePtr]

# export API
alloc = lib.extarray_alloc
free = lib.extarray_free
getpointer = lib.extarray_getpointer
getnbytes = lib.extarray_getnbytes
getrefcount = lib.extarray_getrefcount
make_meminfo = lib.extarray_make_meminfo
acquire = lib.extarray_acquire
meminfo_gethandle = lib.extarray_meminfo_gethandle
