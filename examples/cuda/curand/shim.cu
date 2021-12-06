// Shim functions for calling cuRAND from Numba functions.
//
// Numba's ABI expects that:
//
// - The return value is used to indicate whether a Python exception occurred
//   during function execution. This does not happen in C/C++ kernels, so we
//   always return 0.
// - The result returned to Numba is passed as a pointer in the first parameter.
//   For void functions (such as init_states()), a parameter is passed, but is
//   unused.

#include <curand_kernel.h>

extern "C"
__device__ int init_states(int* return_value, curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);

    // Indicate that no exception occurred
    return 0;
}

extern "C"
__device__ int generate_results(unsigned int* result, curandState *state, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;

    // Copy state to local memory for efficiency
    curandState localState = state[id];

    // Generate pseudo-random unsigned ints
    for(int i = 0; i < n; i++) {
        x = curand(&localState);

        // Check if low bit set
        if(x & 1) {
            count++;
        }
    }
    // Copy state back to global memory
    state[id] = localState;

    // Store results
    *result = count;

    // Indicate that no exception occurred
    return 0;
}
