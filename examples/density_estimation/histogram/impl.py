import numpy as np

def input_generator():
    for nbins in [10, 1000]:
        for dtype in [np.float32, np.float64]:
            for size in [1000, 10000, 100000, 300000, 3000000]:
                category = ('bins%d' % nbins, np.dtype(dtype).name)
                input_array = np.random.normal(loc=0.0, scale=5.0, size=size).astype(dtype)
                yield dict(category=category, x=size, input_args=(input_array, nbins), input_kwargs={})

#### BEGIN: numpy
import numpy as np

def numpy_histogram(a, bins):
    return np.histogram(a, bins)
#### END: numpy

#### BEGIN: numba
import numba
import numpy as np


@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges
#### END: numba


def validator(input_args, input_kwargs, impl_output):
    actual_hist, actual_edges = impl_output
    expected_hist, expected_edges = numpy_histogram(*input_args, **input_kwargs)
    np.testing.assert_allclose(expected_edges, actual_edges, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(expected_hist, actual_hist)
