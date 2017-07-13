import numba

@numba.vectorize(['(int16, int16)','(float32, float32)', '(float64, float64)'], target='cuda')
def numba_zero_suppression(values, threshold):
    if abs(values) >= threshold:
        return values
    else:
        return 0.0
