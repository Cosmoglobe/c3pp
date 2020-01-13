import math
import sys
import os
import time
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


#######################
# ACTUAL MODULES HERE #
#######################


def h5handler(flags, command):
    filename = str(flags[0])
    signal = str(flags[1])
    min = int(flags[2])
    max = int(flags[3])
    outname = flags[-1]
    l = max - min

    # Check if you want to output a map
    map = True if "fits" in outname[-4:] else False

    import h5py

    dats = []
    with h5py.File(filename, "r") as f:
        for sample in range(min, max + 1):

            # Get sample number with leading zeros
            s = str(sample).zfill(6)

            # Get data from hdf
            data = f[s + "/" + signal][()]
            # Smooth every sample if calculating std.
            if "-smooth" in flags and command == np.std and map:
                print("--- Smoothing sample {} ---".format(sample))
                fwhm = arcmin2rad(float(get_key(flags, "-smooth")))
                data = hp.sphtfunc.smoothing(data, fwhm=fwhm, pol=False)
            # Append sample to list
            dats.append(data)

    # Convert list to array
    dats = np.array(dats)

    # Calculate std or mean
    outdata = command(dats, axis=0)

    # Smoothing can be done after for np.mean
    if "-smooth" in flags and command == np.mean and map:
        fwhm = arcmin2rad(get_key(flags, "-smooth"))
        outdata = hp.sphtfunc.smoothing(outdata, fwhm=fwhm, pol=True)

    # Outputs fits map if output name is .fits
    if map:
        hp.write_map(outname, outdata, overwrite=True)
    else:
        np.savetxt(outname, outdata)


def mean(flags):
    h5handler(flags, np.mean)


def stddev(flags):
    h5handler(flags, np.std)


def plot(flags):
    from c3postproc.plotter import Plotter

    Plotter(flags)


def sigma_l2fits(flags, save=True):
    filename = str(flags[0])
    path = "cmb/sigma_l"
    nchains = int(flags[-3])
    burnin = int(flags[-2])
    outname = str(flags[-1])
    if len(flags) == 5:
        path = str(flags[1])

    import h5py

    for nc in range(1, nchains + 1):
        with h5py.File(filename + "_c" + str(nc).zfill(4) + ".h5", "r") as f:
            print("Reading HDF5 file: " + filename + " ...")
            groups = list(f.keys())
            print()
            print("Reading " + str(len(groups)) + " samples from file.")

            if nc == 1:
                dset = np.zeros((len(groups) + 1, 1, len(f[groups[0] + "/" + path]), len(f[groups[0] + "/" + path][0])))
                nspec = len(f[groups[0] + "/" + path])
                lmax = len(f[groups[0] + "/" + path][0]) - 1
                nsamples = len(groups)
            else:
                dset = np.append(dset, np.zeros((nsamples + 1, 1, nspec, lmax + 1)), axis=1)
            print(np.shape(dset))

            print(
                "Found: \
            \npath in the HDF5 file : "
                + path
                + " \
            \nnumber of spectra :"
                + str(nspec)
                + "\nlmax: "
                + str(lmax)
            )

            for i in range(nsamples):
                for j in range(nspec):
                    dset[i + 1, nc - 1, j, :] = np.asarray(f[groups[i] + "/" + path][j][:])

    ell = np.arange(lmax + 1)
    for nc in range(1, nchains + 1):
        for i in range(1, nsamples + 1):
            for j in range(nspec):
                dset[i, nc - 1, j, :] = dset[i, nc - 1, j, :] * ell[:] * (ell[:] + 1.0) / 2.0 / np.pi
    dset[0, :, :, :] = nsamples - burnin

    if save:
        import fitsio

        print("Dumping fits file: " + outname + " ...")
        dset = np.asarray(dset, dtype="f4")
        fits = fitsio.FITS(outname, mode="rw", clobber=True, verbose=True)
        h_dict = [
            {"name": "FUNCNAME", "value": "Gibbs sampled power spectra", "comment": "Full function name"},
            {"name": "LMAX", "value": lmax, "comment": "Maximum multipole moment"},
            {"name": "NUMSAMP", "value": nsamples, "comment": "Number of samples"},
            {"name": "NUMCHAIN", "value": nchains, "comment": "Number of independent chains"},
            {"name": "NUMSPEC", "value": nspec, "comment": "Number of power spectra"},
        ]
        fits.write(dset[:, :, :, :], header=h_dict, clobber=True)
        fits.close()
    return dset


def dlbin2dat(flags):
    filename = str(flags[0])
    signal = "cmb/Dl"
    min = int(flags[1])
    max = int(flags[2])
    binfile = flags[3]

    import h5py

    dats = []
    with h5py.File(filename, "r") as f:
        for sample in range(min, max + 1):
            # Get sample number with leading zeros
            s = str(sample).zfill(6)

            # Get data from hdf
            data = f[s + "/" + signal][()]
            # Append sample to list
            dats.append(data)
    dats = np.array(dats)

    binned_data = {}
    possible_signals = ["TT", "EE", "BB", "TE", "EB", "TB"]
    with open(binfile) as f:
        next(f)  # Skip first line
        for line in f.readlines():
            line = line.split()
            signal = line[0]
            if signal not in binned_data:
                binned_data[signal] = []
            signal_id = possible_signals.index(signal)
            lmin = int(line[1])
            lmax = int(line[2])
            ellcenter = lmin + (lmax - lmin) / 2
            # Saves (ellcenter, lmin, lmax, Dl_mean, Dl_stddev) over samples chosen
            binned_data[signal].append(
                [
                    ellcenter,
                    lmin,
                    lmax,
                    np.mean(dats[:, signal_id, lmin], axis=0),
                    np.std(dats[:, signal_id, lmin], axis=0),
                ]
            )

    header = "{:22} {:24} {:24} {:24} {:24}".format("l", "lmin", "lmax", "Dl", "stddev")
    for signal in binned_data.keys():
        np.savetxt("Dl_" + signal + "_binned.dat", binned_data[signal], header=header)


def h5map2fits(flags, save=True):
    import h5py

    h5file = str(flags[0])
    dataset = str(flags[1])
    with h5py.File(h5file, "r") as f:
        maps = f[dataset][()]
        lmax = f[dataset[:-4] + "_lmax"][()]  # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_map", "")
    if save:
        hp.write_map(outfile + "_n" + str(nside) + ".fits", maps, overwrite=True)
    return maps, nside, lmax, outfile


def alm2fits(flags, save=True):
    import h5py

    h5file = str(flags[0])
    dataset = str(flags[1])
    nside = int(flags[2])  # Output nside

    with h5py.File(h5file, "r") as f:
        alms = f[dataset][()]
        lmax = f[dataset[:-4] + "_lmax"][()]  # Get lmax from h5

    if "-lmax" in flags:
        lmax_ = int(get_key(flags, "-lmax"))
        if lmax_ > lmax:
            print("lmax larger than data allows: ", lmax)
            print("Please chose a value smaller than this")
        else:
            lmax = lmax_
        mmax = lmax
    else:
        mmax = lmax

    if "-fwhm" in flags:
        fwhm = float(get_key(flags, "-fwhm"))
    else:
        fwhm = 0.0

    hehe = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
    print("Setting lmax to ", lmax, "hehe: ", hehe, "datashape: ", alms.shape)

    alms_unpacked = unpack_alms(alms, lmax)  # Unpack alms
    maps = hp.sphtfunc.alm2map(alms_unpacked, nside, lmax=lmax, mmax=mmax, fwhm=arcmin2rad(fwhm))

    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_alm", "")
    if save:
        outfile += "_{}arcmin".format(str(int(fwhm))) if "-fwhm" in flags else ""
        hp.write_map(outfile + "_n" + str(nside) + "_lmax{}".format(lmax) + ".fits", maps, overwrite=True)
    return maps, nside, lmax, fwhm, outfile


#######################
# HELPFUL TOOLS BELOW #
#######################


def unpack_alms(maps, lmax):
    """
    Create lm pairs here (same as commander)
    """

    mind = []
    lm = []
    idx = 0
    # lm pairs where m = 0
    mind.append(idx)
    for l in range(0, lmax + 1):
        lm.append((l, 0))
        idx += 1
    # rest of lm pairs
    for m in range(1, lmax + 1):
        mind.append(idx)
        for l in range(m, lmax + 1):
            lm.append((l, m))
            # lm.append((l,-m))
            idx += 1
    # print(lm[:50])
    """
    unpack data here per l,m pair
    """
    alms = [[], [], []]
    for l, m in lm:
        # ind = hp.Alm.getidx(lmax, l, m)
        # if m < 0:
        #    continue
        if m == 0:
            idx = mind[m] + l
            idx = l ** 2 + l + m
            for pol in range(3):
                a_lm = complex(maps[pol, idx], 0.0)
                alms[pol].append(a_lm)
        else:
            idx = mind[abs(m)] + 2 * (l - abs(m))
            idx = l ** 2 + l + m
            idx2 = idx + 1

            for pol in range(3):
                a_lm = complex(maps[pol, idx], maps[pol, idx2]) / np.sqrt(2)
                alms[pol].append(a_lm)
                # alms2[pol,ind] = a_lm

    alms2 = np.array(alms, dtype=np.complex128)
    """
    hehe = int(lmax * (2 * lmax + 1 - lmax) / 2 + lmax + 1) 
    alms2 = np.zeros((3,hehe), dtype=np.complex128)
   
    for j, (l, m) in enumerate(lm):
        ind = l**2 + l + m
        alms2[:,j] = alms[:,idx]
    """
    return alms2


def get_key(flags, keyword):
    return flags[flags.index(keyword) + 1]


def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600


"""

lm2 = lm
mind = []
lm = []
idx = 0
for m in range(lmax+1):
    mind.append(idx)
    if m == 0:
        for l in range(m, lmax+1):
            lm.append((l,m))
            idx += 1
    else:
        for l in range(m,lmax+1):
            lm.append((l,m))
            idx +=1
            
            lm.append((l,-m))
            idx +=1

lm = np.zeros((2,22801))
mind = np.zeros(lmax+1)
ind = 0
for m in range(lmax+1):
    mind[m] = ind
    if m == 0:
        for l in range(m, lmax+1):
            lm[:,ind] = (l,m)
            ind                           = ind+1
    else:
        for l in range(m, lmax+1):
        lm[:,ind] = (l,m)
        ind                           = ind+1
        lm[:,ind] = (l,-m)
        ind                           = ind+1
print(lm.shape)


  alms1 =[[],[],[]] 
    for l, m in lm:
        if m<0:
            continue
        idx = lm2i(l, m,mind)
        if m == 0:
            for pol in range(3):
                alms1[pol].append( complex( maps[pol,idx], 0.0 ) )
        else:
            idx2 = lm2i(l,-m,mind)
            for pol in range(3):
                alms1[pol].append( 1/np.sqrt(2)*complex(maps[pol,idx], maps[pol,idx2]) )




def lm2i(l,m,mind):
    if m == 0:
        i = mind[int(m)] + l
    else:
        i = mind[int(abs(m))] + 2*(l-abs(m))
        if (m < 0):
           i = i+1
    return int(i)
"""
