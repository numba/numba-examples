// Shim functions for calling cuRAND from Numba functions.
//
// Numba's ABI expects that:
//
// - The return value is used to indicate whether a Python exception occurred
//   during function execution. This does not happen in C/C++ kernels, so we
//   always return 0.
// - The result returned to Numba is passed as a pointer in the first parameter.
//   For void functions (such as curand_init()), a parameter is passed, but is
//   unused.

#include <curand_kernel.h>

extern "C"
__device__ int _numba_curand_init(
    int* numba_return_value,
    unsigned long long seed,
    unsigned long long sequence,
    unsigned long long offset,
    curandState *state)
{
    curand_init(seed, sequence, offset, state);

    return 0;
}

extern "C"
__device__ unsigned int _numba_curand(
    int* numba_return_value,
    curandState *state)
{
  *numba_return_value = curand(state);

  return 0;
}
