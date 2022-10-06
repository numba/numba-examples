import numpy as np

#### BEGIN: numpy
import math

def mandelbrot(iters, pixels, bbox, width, max_iter):
    for y in range(width):
        for x in range(width):
            c0 = complex(bbox[0] + (bbox[2]-bbox[0])*x/width, 
                         bbox[1] + (bbox[3]-bbox[1])*y/width) 
            c = 0
            i = 1
            while i < max_iter:
                if abs(c) > 2:  
                    log_iter = math.log(i) 
                    pixels[y, x, 0] = 255*(1+math.cos(3.32*log_iter))//2
                    pixels[y, x, 1] = 255*(1+math.cos(0.774*log_iter))//2
                    pixels[y, x, 2] = 255*(1+math.cos(0.412*log_iter))//2                    
                    break
                c = c * c + c0
                i += 1
            iters[y, x] = i                
    return iters, pixels
## END: numpy

WIDTH = 600
PLANE = (-2.0, -1.5, 1.0, 1.5)

def validator(input_args, input_kwargs, impl_output):
    actual_iters, actual_pixels  = impl_output
    expected_iters, expected_pixels = mandelbrot(*input_args, **input_kwargs)
    np.testing.assert_allclose(expected_iters, actual_iters)
    np.testing.assert_allclose(expected_pixels, actual_pixels)

def make_arrays(width):
    iters = np.zeros((width, width), dtype=np.uint16)
    pixels = np.zeros((width, width, 3), dtype=np.uint8)
    return iters, pixels

def input_generator():
    for maxiter in [100, 200, 500, 1000]:
        iters, pixels = make_arrays(WIDTH)
        yield dict(category=("",),
                x=maxiter,
                input_args=(iters, pixels, PLANE, WIDTH, maxiter),
                input_kwargs={})
