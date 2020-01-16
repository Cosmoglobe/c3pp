import numba
import numpy as np

#######################
# HELPFUL TOOLS BELOW #
#######################


@numba.njit(cache=True, fastmath=True)  # Speeding up by a lot!
def unpack_alms(maps, lmax):
    print("Unpacking alms")
    mmax = lmax
    # hehe is length of target alms
    hehe = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
    alms = np.zeros((maps.shape[0], hehe), dtype=np.complex128)

    # Unpack alms as output by commander
    for sig in range(3):
        i = 0
        for l in range(lmax + 1):
            j_real = l ** 2 + l
            alms[sig, i] = complex(maps[sig, j_real], 0.0)
            i += 1

        for m in range(1, lmax + 1):
            for l in range(m, lmax + 1):
                j_real = l ** 2 + l + m
                j_comp = l ** 2 + l - m

                alms[sig, i] = complex(maps[sig, j_real], maps[sig, j_comp]) / np.sqrt(2.0)

                i += 1
    return alms


def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600
