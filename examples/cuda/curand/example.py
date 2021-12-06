# Demonstration of calling cuRAND functions from Numba kernels. Shim functions
# in a .cu file are used to access the cuRAND functions from Numba. This is
# based on the cuRAND device API example in:
# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
#
# The result produced by this example agrees with the documentation example.
# E.g. on a particular configuration:
#
# Output from this example:                  0.4999931156635
# Output from cuRAND documentation example:  0.4999931156635

from numba import cuda
import numpy as np

# Various parameters

threads = 64
blocks = 64
nthreads = blocks * threads

sample_count = 10000
repetitions = 50


# cuRAND state type - mirrors the state defined in curand_kernel.h
state_fields = [
    ('d', np.int32),
    ('v', np.int32, 5),
    ('boxmuller_flag', np.int32),
    ('boxmuller_flag_double', np.int32),
    ('boxmuller_extra', np.float32),
    ('boxmuller_extra_double', np.float64),
]

curandState = np.dtype(state_fields, align=True)


# Forward declarations of shim functions in the .cu source

init_states = cuda.declare_device('init_states', 'void(uint64)')
generate_results = cuda.declare_device(
    'generate_results', 'int32(uint64, int32)')


# Kernels that call shim functions. Shim functions take a pointer to the data
# rather than a Numba array type.

@cuda.jit(link=['shim.cu'])
def setup_curand(states_ptr):
    init_states(states_ptr)


@cuda.jit(link=['shim.cu'])
def count_low_bits(states_ptr, sample_count, results):
    i = cuda.grid(1)
    results[i] += generate_results(states_ptr, sample_count)


# Create state on the device. The CUDA Array Interface provides a convenient
# way to get the pointer needed for the shim functions.

state = cuda.device_array(nthreads, dtype=curandState)
states_ptr = state.__cuda_array_interface__['data'][0]

results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))


# Initialise cuRAND state

setup_curand[blocks, threads](states_ptr)


# Run random sampling kernel

for i in range(repetitions):
    count_low_bits[blocks, threads](states_ptr, sample_count, results)


# Collect the results and summarize them. This could have been done on device,
# but the corresponding CUDA C++ sample does it on the host, and we're
# following that example.

host_results = results.copy_to_host()

total = 0
for i in range(nthreads):
    total += host_results[i]

# Use float32 to show an exact match between this and the cuRAND documentation
# example
fraction = (np.float32(total) /
            np.float32(nthreads * sample_count * repetitions))

print(f"Fraction with low bit set was {fraction:17.13f}")
