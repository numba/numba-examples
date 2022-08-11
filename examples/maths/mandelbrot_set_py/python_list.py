import math

#### BEGIN: python list
def mandelbrot(bbox, width, max_iter):     
    pixels = [[(0, 0, 0) for j in range(width)] for i in range(width)]
    for y in range(width):
        for x in range(width):
            c0 = complex(bbox[0] + (bbox[2]-bbox[0])*x/width, 
                         bbox[1] + (bbox[3]-bbox[1])*y/width) 
            c = 0
            for i in range(1, max_iter): 
                if abs(c) > 2: 
                    log_iter = math.log(i) 
                    pixels[y][x] = (int(255*(1+math.cos(3.32*log_iter))/2), 
                                    int(255*(1+math.cos(0.774*log_iter))/2), 
                                    int(255*(1+math.cos(0.412*log_iter))/2))
                    break
                c = c * c + c0
    return pixels 
#### END: python list


def validator(input_args, input_kwargs, impl_output):
    actual_output = impl_output
    expected_output = mandelbrot(*input_args, **input_kwargs)
    assert actual_output == expected_output

def input_generator():
    for maxiter in [100, 200, 500, 1000]:
        width = 600
        plane = (-2.0, -1.5, 1.0, 1.5)
        yield dict(category=("",),
                   x=maxiter,
                   input_args=(plane, width, maxiter),
                   input_kwargs={})

