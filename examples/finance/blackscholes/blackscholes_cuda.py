from __future__ import print_function

import math
import time

import numpy as np

from numba import cuda


@cuda.jit(device=True)
def cnd_cuda(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cuda.jit
def black_scholes_cuda_kernel(callResult, putResult, S, X,
                       T, R, V):
    #    S = stockPrice
    #    X = optionStrike
    #    T = optionYears
    #    R = Riskfree
    #    V = Volatility
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_cuda(d1)
    cndd2 = cnd_cuda(d2)

    expRT = math.exp((-1. * R) * T[i])
    callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
    putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


def black_scholes_cuda(stockPrice, optionStrike,
                        optionYears, Riskfree, Volatility):

    blockdim = 1024, 1
    griddim = int(math.ceil(float(len(stockPrice))/blockdim[0])), 1

    stream = cuda.stream()

    d_callResult = cuda.device_array_like(stockPrice, stream)
    d_putResult = cuda.device_array_like(stockPrice, stream)
    d_stockPrice = cuda.to_device(stockPrice, stream)
    d_optionStrike = cuda.to_device(optionStrike, stream)
    d_optionYears = cuda.to_device(optionYears, stream)

    black_scholes_cuda_kernel[griddim, blockdim, stream](
            d_callResult, d_putResult, d_stockPrice, d_optionStrike,
            d_optionYears, Riskfree, Volatility)
    callResult = d_callResult.copy_to_host(stream=stream)
    putResult= d_putResult.copy_to_host(stream=stream)
    stream.synchronize()

    return callResult, putResult
