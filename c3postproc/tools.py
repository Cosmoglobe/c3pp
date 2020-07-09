import numba
import numpy as np

#######################
# HELPFUL TOOLS BELOW #
#######################


@numba.njit(cache=True, fastmath=True)  # Speeding up by a lot!
def unpack_alms(maps, lmax):
    #print("Unpacking alms")
    mmax = lmax
    nmaps = len(maps)
    # Nalms is length of target alms
    Nalms = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
    alms = np.zeros((nmaps, Nalms), dtype=np.complex128)

    # Unpack alms as output by commander
    for sig in range(nmaps):
        i = 0
        for l in range(lmax + 1):
            j_real = l ** 2 + l
            alms[sig, i] = complex(maps[sig, j_real], 0.0)
            i += 1

        for m in range(1, lmax + 1):
            for l in range(m, lmax + 1):
                j_real = l ** 2 + l + m
                j_comp = l ** 2 + l - m

                alms[sig, i] = complex(maps[sig, j_real], maps[sig, j_comp],) / np.sqrt(2.0)

                i += 1
    return alms


def alm2fits_tool(input, dataset, nside, lmax, fwhm, save=True):
    import h5py
    import healpy as hp

    with h5py.File(input, "r") as f:
        alms = f[dataset][()]
        lmax_h5 = f[f"{dataset[:-3]}lmax"][()]  # Get lmax from h5

    if lmax:
        # Check if chosen lmax is compatible with data
        if lmax > lmax_h5:
            print(
                "lmax larger than data allows: ", lmax_h5,
            )
            print("Please chose a value smaller than this")
    else:
        # Set lmax to default value
        lmax = lmax_h5
    mmax = lmax

    alms_unpacked = unpack_alms(alms, lmax)  # Unpack alms

    print("Making map from alms")
    maps = hp.sphtfunc.alm2map(alms_unpacked, nside, lmax=lmax, mmax=mmax, fwhm=arcmin2rad(fwhm), pixwin=True,)

    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_alm", "")
    if save:
        outfile += f"_{str(int(fwhm))}arcmin" if fwhm > 0.0 else ""
        hp.write_map(
            outfile + f"_n{str(nside)}_lmax{lmax}.fits", maps, overwrite=True,
        )
    return maps, nside, lmax, fwhm, outfile


def h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, command,):
    # Check if you want to output a map
    import h5py
    import healpy as hp
    from tqdm import tqdm

    print()
    print("{:-^48}".format(f" {dataset} calculating {command.__name__} "))
    print("{:-^48}".format(f" nside {nside}, {fwhm} arcmin smoothing "))

    if dataset.endswith("map"):
        type = "map"
    elif dataset.endswith("alm"):
        type = "alm"
    elif dataset.endswith("sigma"):
        type = "sigma"
    else:
        print(f"Type {type} not recognized")
        sys.exit()

    dats = []
    maxnone = True if max == None else False  # set length of keys for maxchains>1
    for c in range(1, maxchain + 1):
        filename = input.replace("c0001", "c" + str(c).zfill(4))
        with h5py.File(filename, "r") as f:
            if maxnone:
                # If no max is specified, chose last sample
                max = len(f.keys()) - 1

            print("{:-^48}".format(f" Samples {min} to {max} chain {c}"))

            for sample in tqdm(range(min, max + 1), ncols=80):
                # Identify dataset
                # alm, map or (sigma_l, which is recognized as l)

                # Unless output is ".fits" or "map", don't convert alms to map.
                alm2map = True if output.endswith((".fits", "map")) else False

                # HDF dataset path formatting
                s = str(sample).zfill(6)

                # Sets tag with type
                tag = f"{s}/{dataset}"
                #print(f"Reading c{str(c).zfill(4)} {tag}")

                # Check if map is available, if not, use alms.
                # If alms is already chosen, no problem
                try:
                    data = f[tag][()]
                    if len(data[0]) == 0:
                        tag = f"{tag[:-3]}map"
                        print(f"WARNING! No {type} data found, switching to map.")
                        data = f[tag][()]
                        type = "map"
                except:
                    print(f"Found no dataset called {dataset}")
                    print(f"Trying alms instead {tag}")
                    try:
                        # Use alms instead (This takes longer and is not preferred)
                        tag = f"{tag[:-3]}alm"
                        type = "alm"
                        data = f[tag][()]
                    except:
                        print("Dataset not found.")

                # If data is alm, unpack.
                if type == "alm":
                    lmax_h5 = f[f"{tag[:-3]}lmax"][()]
                    data = unpack_alms(data, lmax_h5)  # Unpack alms

                if data.shape[0] == 1:
                    # Make sure its interprated as I by healpy
                    # For non-polarization data, (1,npix) is not accepted by healpy
                    data = data.ravel()

                # If data is alm and calculating std. Bin to map and smooth first.
                if type == "alm" and command == np.std and alm2map:
                    #print(f"#{sample} --- alm2map with {fwhm} arcmin, lmax {lmax_h5} ---")
                    data = hp.alm2map(data, nside=nside, lmax=lmax_h5, fwhm=arcmin2rad(fwhm), pixwin=True,verbose=False,)

                # If data is map, smooth first.
                elif type == "map" and fwhm > 0.0 and command == np.std:
                    #print(f"#{sample} --- Smoothing map ---")
                    data = hp.sphtfunc.smoothing(data, fwhm=arcmin2rad(fwhm),verbose=False,)

                # Append sample to list
                dats.append(data)

    # Convert list to array
    dats = np.array(dats)
    # Calculate std or mean
    outdata = command(dats, axis=0)

    # Smoothing afterwards when calculating mean
    if type == "alm" and command == np.mean and alm2map:
        #print(f"# --- alm2map mean with {fwhm} arcmin, lmax {lmax_h5} ---")
        outdata = hp.alm2map(outdata, nside=nside, lmax=lmax_h5, fwhm=arcmin2rad(fwhm), pixwin=True,verbose=False,)

    # Smoothing can be done after for np.mean
    if type == "map" and fwhm > 0.0 and command == np.mean:
        #print(f"--- Smoothing mean map with {fwhm} arcmin,---")
        outdata = hp.sphtfunc.smoothing(outdata, fwhm=arcmin2rad(fwhm), verbose=False,)

    # Outputs fits map if output name is .fits
    if output.endswith(".fits"):
        hp.write_map(output, outdata, overwrite=True)
    elif output.endswith(".dat"):
        np.savetxt(output, outdata)
    else:
        return outdata


def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600
