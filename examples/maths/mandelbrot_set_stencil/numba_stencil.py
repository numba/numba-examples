import numba as nb
import numpy as np

@nb.stencil 
def iterskernel(a, max_iter):
    c = 0
    i = 1
    while i < max_iter:
        if abs(c) > 2:  
            break
        c = c * c + a[0, 0]
        i += 1
    return i

@nb.njit
def mandelbrot(iters, pixels, bbox, width, max_iter):
    arr = [[complex(bbox[0] + (bbox[2]-bbox[0])*x/width, 
                      bbox[1] + (bbox[3]-bbox[1])*y/width) for x in range(width)] 
              for y in range(width)]
    c_arr = np.array(arr)
    iters = iterskernel(c_arr, max_iter)
    pix = iters.reshape((width, width, 1))*np.array([1,1,1])
    pix = np.where(pix == max_iter, 
                   0, 
                   255*(1+np.cos(np.log(pix)*np.array([3.32, 0.774, 0.412])))//2)
    pixels = pix.astype(np.uint8)
    return iters, pixels
