import numpy as np

#### BEGIN: numpy vectorize

@np.vectorize
def get_max_iter(x, y, width, max_iter, r1, i1, r2, i2):
    c0 = complex(r1 + (r2-r1)*x/width, 
                 i1 + (i2-i1)*y/width)  
    c = 0
    for i in range(1, max_iter+1): 
        if abs(c) > 2:  
            break
        c = c * c + c0
    return i

def mandelbrot(bbox, width, max_iter):  
    grid1D = np.arange(0, width)
    xv, yv = np.meshgrid(grid1D, grid1D)  
    r1, i1, r2, i2 = bbox
    iters = get_max_iter(xv, yv, width, max_iter, r1, i1, r2, i2).reshape((width, width, 1))

    pixels = np.where(
                iters == np.array([max_iter]), 
                np.array([0, 0, 0]), 
                255*(1+np.cos(np.log(iters)*np.array([3.32, 0.774, 0.412])))//2)

    iters = iters.reshape((width, width))
    pixels = pixels.astype(np.uint8)
    return iters, pixels
## END: numpy vectorize

def validator(input_args, input_kwargs, impl_output):
    actual_iters, actual_pixels  = impl_output
    expected_iters, expected_pixels = mandelbrot(*input_args, **input_kwargs)
    np.testing.assert_allclose(expected_iters, actual_iters)
    np.testing.assert_allclose(expected_pixels, actual_pixels)

def make_pixels(width):
    pixels = np.zeros((width, width, 3), dtype=np.uint8)
    return pixels

def input_generator():
    for maxiter in [100, 200, 500, 1000]:
        width = 600
        #pixels = make_pixels(width)
        plane = (-2.0, -1.5, 1.0, 1.5)
        yield dict(category=("",),
                   x=maxiter,
                   input_args=(plane, width, maxiter),
                   input_kwargs={})