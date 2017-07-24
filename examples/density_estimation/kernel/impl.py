import numpy as np

def input_generator():
    for dtype in [np.float64]:
        for nsamples in [1000, 10000]:
            sigma = 5.0
            samples = np.random.normal(loc=0.0, scale=sigma, size=nsamples).astype(dtype)
            # For simplicity, initialize bandwidth array with constant using 1D rule of thumb
            bandwidths = np.full_like(samples, 1.06 * nsamples**0.2 * sigma)
            for neval in [10, 1000, 10000]:
                category = ('samples%d' % nsamples, np.dtype(dtype).name)
                eval_points = np.random.normal(loc=0.0, scale=5.0, size=neval).astype(dtype)
                yield dict(category=category, x=neval, input_args=(eval_points, samples, bandwidths), input_kwargs={})

#### BEGIN: numpy
import numpy as np

def numpy_kde(eval_points, samples, bandwidths):
    # This uses a lot of RAM and doesn't scale to larger datasets
    rescaled_x = (eval_points[:, np.newaxis] - samples[np.newaxis, :]) / bandwidths[np.newaxis, :]
    gaussian = np.exp(-0.5 * rescaled_x**2) / np.sqrt(2 * np.pi) / bandwidths[np.newaxis, :]
    return gaussian.sum(axis=1) / len(samples)

#### END: numpy

#### BEGIN: numba
import numba
import numpy as np

@numba.jit(nopython=True)
def gaussian(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

@numba.jit(nopython=True)
def numba_kde(eval_points, samples, bandwidths):
    result = np.zeros_like(eval_points)

    for i, eval_x in enumerate(eval_points):
        for sample, bandwidth in zip(samples, bandwidths):
            result[i] += gaussian((eval_x - sample) / bandwidth) / bandwidth
        result[i] /= len(samples)

    return result
#### END: numba


#### BEGIN: numba_multithread
import numba
import numpy as np

@numba.jit(nopython=True, parallel=True)
def numba_kde_multithread(eval_points, samples, bandwidths):
    result = np.zeros_like(eval_points)

    # SPEEDTIP: Parallelize over evaluation points with prange()
    for i in numba.prange(len(eval_points)):
        eval_x = eval_points[i]
        for sample, bandwidth in zip(samples, bandwidths):
            result[i] += gaussian((eval_x - sample) / bandwidth) / bandwidth
        result[i] /= len(samples)

    return result
#### END: numba_multithread


def validator(input_args, input_kwargs, impl_output):
    actual_y = impl_output
    expected_y = numpy_kde(*input_args, **input_kwargs)
    np.testing.assert_allclose(expected_y, actual_y, rtol=1e-6, atol=1e-6)
