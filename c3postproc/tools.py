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


def alm2fits_tool(input, dataset, nside, lmax, fwhm, save=True):
    import h5py
    import healpy as hp

    with h5py.File(input, "r") as f:
        alms = f[dataset][()]
        lmax_h5 = f[dataset[:-4] + "_lmax"][()]  # Get lmax from h5

    if lmax:
        # Check if chosen lmax is compatible with data
        if lmax > lmax_h5:
            click.echo("lmax larger than data allows: ", lmax_h5)
            click.echo("Please chose a value smaller than this")
    else:
        # Set lmax to default value
        lmax = lmax_h5
    mmax = lmax

    alms_unpacked = unpack_alms(alms, lmax)  # Unpack alms

    click.echo("Making map from alms")
    maps = hp.sphtfunc.alm2map(alms_unpacked, nside, lmax=lmax, mmax=mmax, fwhm=arcmin2rad(fwhm))

    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_alm", "")
    if save:
        outfile += "_{}arcmin".format(str(int(fwhm))) if fwhm > 0.0 else ""
        hp.write_map(outfile + "_n" + str(nside) + "_lmax{}".format(lmax) + ".fits", maps, overwrite=True)
    return maps, nside, lmax, fwhm, outfile


def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600
