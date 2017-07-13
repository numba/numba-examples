import numpy as np

def input_generator():
    for dtype in [np.int16, np.float32, np.float64]:
        for size in [100, 1000, 10000, 50000]:
            name = np.dtype(dtype).name
            input_array = np.random.normal(loc=0.0, scale=5.0, size=size)
            # add a pulse train
            input_array += 50 * np.clip(np.cos(np.linspace(0.0, 1.0, num=size)*np.pi*10), 0, 1.0)
            input_array = input_array.astype(dtype)
            yield dict(category=('%s' % name,), x=size, input_args=(input_array, 8.0), input_kwargs={})

#### BEGIN: numpy
import numpy as np

def numpy_zero_suppression(values, threshold):
    result = np.zeros_like(values)
    selector = np.abs(values) >= threshold
    result[selector] = values[selector]
    return result
#### END: numpy

#### BEGIN: numba_single_thread
import numba
import numpy as np

@numba.jit(nopython=True)
def numba_zero_suppression(values, threshold):
    result = np.zeros_like(values)
    for i in range(values.shape[0]):
        if np.abs(values[i]) >= threshold:
            result[i] = values[i]
    return result
#### END: numba_single_thread


def validator(input_args, input_kwargs, impl_output):
    # We're using the Numpy implementation as the reference
    expected = numpy_zero_suppression(*input_args, **input_kwargs)
    np.testing.assert_array_equal(expected, impl_output)
