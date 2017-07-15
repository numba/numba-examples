import numpy as np
import numba

@numba.njit
def lj_numba_array(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


@numba.njit
def distances_numba_array(cluster):
    # Original: diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
    # Since np.newaxis is not supported, we use reshape to do this
    diff = (cluster.reshape(cluster.shape[0], 1, cluster.shape[1]) -
            cluster.reshape(1, cluster.shape[0], cluster.shape[1]))
    mat = (diff * diff)
    # Original: mat = mat.sum(-1)
    # Since axis argument is not supported, we write the loop out
    out = np.empty(mat.shape[:2], dtype=mat.dtype)
    for i in np.ndindex(out.shape):
        out[i] = mat[i].sum()

    return np.sqrt(out)


@numba.njit
def potential_numba_array(cluster):
    d = distances_numba_array(cluster)
    # Original: dtri = np.triu(d)
    # np.triu is not supported; so write my own loop to clear the
    # lower triangle
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if i > j:
                d[i, j] = 0
    # Original: lj_numba_array(d[d > 1e-6]).sum()
    # d[d > 1e-6] is not supported due to the indexing with boolean
    # array.  Replace with custom loop.
    energy = 0.0
    for v in d.flat:
        if v > 1e-6:
            energy += lj_numba_array(v)
    return energy
