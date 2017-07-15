#### BEGIN: numpy
import numpy as np

def lj_numpy(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


def distances_numpy(cluster):
    diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
    mat = np.sqrt((diff * diff).sum(-1))
    return mat


def potential_numpy(cluster):
    d = distances_numpy(cluster)
    dtri = np.triu(d)
    energy = lj_numpy(dtri[dtri > 1e-6]).sum()
    return energy
#### END: numpy


def validator(input_args, input_kwargs, impl_output):
    actual_output = impl_output
    expected_output = potential_numpy(*input_args, **input_kwargs)
    np.testing.assert_allclose(expected_output, actual_output)


def make_cluster(natoms, radius=20, seed=1981):
    np.random.seed(seed)
    cluster = np.random.normal(0, radius, size=(natoms, 3)) - 0.5
    return cluster


def input_generator():
    # for natoms in [100, 1000, 5000]:
    for natoms in [100, 500, 1000, 1500, 2000]:
        cluster = make_cluster(natoms)
        dtype = np.float64
        yield dict(category=(np.dtype(dtype).name,),
                   x=natoms,
                   input_args=(cluster,),
                   input_kwargs={})
