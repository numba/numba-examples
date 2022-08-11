import math
import numba as nb

@nb.njit
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
