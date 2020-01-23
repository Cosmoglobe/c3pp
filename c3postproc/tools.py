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

                alms[sig, i] = complex(
                    maps[sig, j_real], maps[sig, j_comp]
                ) / np.sqrt(2.0)

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
    maps = hp.sphtfunc.alm2map(
        alms_unpacked, nside, lmax=lmax, mmax=mmax, fwhm=arcmin2rad(fwhm)
    )

    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_alm", "")
    if save:
        outfile += f"_{str(int(fwhm))}arcmin" if fwhm > 0.0 else ""
        hp.write_map(
            outfile + f"_n{str(nside)}_lmax{lmax}.fits", maps, overwrite=True
        )
    return maps, nside, lmax, fwhm, outfile


def h5handler(input, dataset, min, max, smooth, output, nside, command):
    # Check if you want to output a map

    import h5py
    import healpy as hp
    dats = []
    with h5py.File(input, "r") as f:
        if max == None:
            max = len(f.keys)-1

        for sample in range(min, max + 1):
            # Get sample number with leading zeros
            s = str(sample).zfill(6)
            # Get data from hdf
            print("Reading ", s + "/" + dataset)
            doalm = False
            try:
                # Check if map is available, if not, use alms.
                data = f[s + "/" + dataset][()]
            except: 
                print(f"Found no dataset called {dataset}")
                print(f'Trying alms instead {dataset[:-4] + "_alm"}')
                try:
                    # Use alms instead (This takes longer and is not preferred)
                    data = f[s + "/" + dataset[:-4] + "_alm"][()]
                    doalm = True
                except:
                    print("Dataset not found.")

            # Smooth every sample if calculating std.
            if smooth != None and command == np.std and output.endswith(".fits"):
                click.echo(f"--- Smoothing sample {sample} ---")
                fwhm = arcmin2rad(smooth)
                data = hp.sphtfunc.smoothing(data, fwhm=fwhm, pol=False)
            # Append sample to list
            dats.append(data)

    # Convert list to array
    dats = np.array(dats)

    # Calculate std or mean
    outdata = command(dats, axis=0)

    # Smoothing can be done after for np.mean
    if smooth != None and command == np.mean and output.endswith(".fits"):
        fwhm = arcmin2rad(smooth)
        outdata = hp.sphtfunc.smoothing(outdata, fwhm=fwhm, pol=True)

    # Outputs fits map if output name is .fits
    if output != None:
        if output.endswith(".fits"):
            hp.write_map(output, outdata, overwrite=True)
        else:
            np.savetxt(output, outdata)
    else:
        # If input data is alms, bin to maps. 
        if dataset.endswith("alm") or doalm:
            lmax_h5 = f[dataset[:-4] + "_lmax"][()]
            outdata = hp.alm2map(outdata, nside=nside, lmax=lmax_h5)

        return outdata

def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600
