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
#
# Note that this example requires an installation of the CUDA toolkit 11.0 or
# later, because NVRTC will use the include files from the installed CUDA
# toolkit.

import sys

try:
    from cuda import cuda as cuda_driver  # noqa: F401
    from numba import config
    config.CUDA_USE_NVIDIA_BINDING = True
except ImportError:
    print("This example requires the NVIDIA CUDA Python Bindings. "
          "Please see https://nvidia.github.io/cuda-python/install.html for "
          "installation instructions.")
    sys.exit(1)

from numba import cuda
from numba_curand import (curand_init, curand, curand_state_arg_handler,
                          CurandStates)
import numpy as np

# Various parameters

threads = 64
blocks = 64
nthreads = blocks * threads

sample_count = 10000
repetitions = 50


# State initialization kernel

@cuda.jit(link=['shim.cu'], extensions=[curand_state_arg_handler])
def setup(states):
    i = cuda.grid(1)
    curand_init(1234, i, 0, states, i)


# Random sampling kernel - computes the fraction of numbers with low bits set
# from a random distribution.

@cuda.jit(link=['shim.cu'], extensions=[curand_state_arg_handler])
def count_low_bits_native(states, sample_count, results):
    i = cuda.grid(1)
    count = 0

    # Copy state to local memory
    # XXX: TBC

    # Generate pseudo-random numbers
    for sample in range(sample_count):
        x = curand(states, i)

        # Check if low bit set
        if(x & 1):
            count += 1

    # Copy state back to global memory
    # XXX: TBC

    # Store results
    results[i] += count


# Create state on the device. The CUDA Array Interface provides a convenient
# way to get the pointer needed for the shim functions.

# Initialise cuRAND state

states = CurandStates(nthreads)
setup[blocks, threads](states)

# Run random sampling kernel

results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

for i in range(repetitions):
    count_low_bits_native[blocks, threads](
        states, sample_count, results)


# Collect the results and summarize them. This could have been done on
# device, but the corresponding CUDA C++ sample does it on the host, and
# we're following that example.

host_results = results.copy_to_host()

total = 0
for i in range(nthreads):
    total += host_results[i]

# Use float32 to show an exact match between this and the cuRAND
# documentation example
fraction = (np.float32(total) /
            np.float32(nthreads * sample_count * repetitions))

print(f"Fraction with low bit set was {fraction:17.13f}")
