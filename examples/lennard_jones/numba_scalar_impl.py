import numba

@numba.njit
def lj_numba_scalar(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


@numba.njit
def distance_numba_scalar(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx * dx + dy * dy + dz * dz) ** 0.5

    return r


@numba.njit
def potential_numba_scalar(cluster):
    energy = 0.0
    for i in range(len(cluster)-1):
        for j in range(i + 1, len(cluster)):
            r = distance_numba_scalar(cluster[i], cluster[j])
            e = lj_numba_scalar(r)
            energy += e

    return energy
