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
            print("lmax larger than data allows: ", lmax_h5)
            print("Please chose a value smaller than this")
    else:
        # Set lmax to default value
        lmax = lmax_h5
    mmax = lmax

    alms_unpacked = unpack_alms(alms, lmax)  # Unpack alms

    print("Making map from alms")
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


def h5handler(input, dataset, min, max, output, fwhm, nside, command):
    # Check if you want to output a map
    import h5py
    import healpy as hp

    # Identify dataset
    if dataset.endswith("alm"):
        type = "alm"
    elif dataset.endswith("map"):
        type = "map"
    elif dataset.endswith("sigma_l"):
        type = "sigma"
    else:
        print(f"Dataset {dataset} not recognized")
        sys.exit()
    
    # Should potential alms be converted to map
    if output.endswith(".fits"):
        alm2map = True
    elif output.endswith(".dat"):
        alm2map = False
    elif output.endswith("map"):
        alm2map = True
    else:
        alm2map=False
    

    dats = []
    with h5py.File(input, "r") as f:
        if max == None:
            # If no max is specified, chose last sample
            max = len(f.keys())-1

        print()
        print(f"--- Calculating {command.__name__}                ---")
        print(f"--- of {dataset} from {min} to {max}         ---")
        print(f"--- at nside {nside} with {fwhm} arcmin smoothing ---")

        for sample in range(min, max + 1):
            # HDF dataset path formatting
            s = str(sample).zfill(6)
            tag = f"{s}/{dataset}"
            print(f"Reading {tag}")

            # Check if map is available, if not, use alms.
            # If alms is already chosen, no problem
            try:
                data = f[tag][()]
            except: 
                print(f"Found no dataset called {dataset}")
                print(f'Trying alms instead {tag}')
                try:
                    # Use alms instead (This takes longer and is not preferred)
                    tag = f"{s}/{dataset[:-4]}_alm"
                    type = "alm"
                    data = f[tag][()]
                except:
                    print("Dataset not found.")

            # If input data is alms, and output is not .dat bin to map.
            # IF output name is .fits, alms will be binned to map.
            # TODO This might be inefficient at some points?
            if type == "alm" and alm2map:
                lmax_h5 = f[dataset[:-4] + "_lmax"][()]
                alms_unpacked = unpack_alms(data, lmax_h5)  # Unpack alms
                data = hp.alm2map(alms_unpacked, nside=nside, lmax=lmax_h5)
                type = "map"
            
            # Smooth every sample if calculating std.
            if fwhm > 0.0 and command == np.std and type == "map":
                print(f"--- Smoothing sample {sample} ---")
                data = hp.sphtfunc.smoothing(data, fwhm=arcmin2rad(fwhm))
            # Append sample to list
            dats.append(data)

    # Convert list to array
    dats = np.array(dats)
    # Calculate std or mean
    outdata = command(dats, axis=0)
    # Smoothing can be done after for np.mean
    if fwhm > 0.0 and command == np.mean and type == "map":
        outdata = hp.sphtfunc.smoothing(outdata, fwhm=arcmin2rad(fwhm))

    # Outputs fits map if output name is .fits
    if output.endswith(".fits"):
        hp.write_map(output, outdata, overwrite=True)
    elif output.endswith(".dat"):
        np.savetxt(output, outdata)
    else:
        return outdata

def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600
