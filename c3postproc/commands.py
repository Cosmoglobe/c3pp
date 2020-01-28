import time
import os
import numpy as np
import click
from c3postproc.tools import *


@click.group()
def commands():
    pass


@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("min", nargs=1, type=click.INT)
@click.argument("max", nargs=1, type=click.INT)
@click.argument("output", type=click.STRING)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.STRING, help="Nside for alm binning")
def mean(
    input, dataset, min, max, output, fwhm, nside,
):
    """
    Calculates the mean over sample range from .h5 file.\n
    ex. chains_c0001.h5 dust/amp_map 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40\n
    If output name is set to .dat, data will not be converted to map.
    """
    if dataset.endswith("alm") and nside == None:
        click.echo("Please specify nside when handling alms.")
        sys.exit()

    h5handler(input, dataset, min, max, output, fwhm, nside, np.mean)


@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("min", nargs=1, type=click.INT)
@click.argument("max", nargs=1, type=click.INT)
@click.argument("output", type=click.STRING)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.STRING, help="Nside for alm binning")
def stddev(
    input, dataset, min, max, output, fwhm, nside,
):
    """
    Calculates the stddev over sample range from .h5 file.\n
    ex. chains_c0001.h5 dust/amp_map 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40\n
    If output name is set to .dat, data will not be converted to map.
    """
    if dataset.endswith("alm") and nside == None:
        click.echo("Please specify nside when handling alms.")
        sys.exit()

    h5handler(input, dataset, min, max, output, fwhm, nside, np.std)


@commands.command()
@click.argument("input", type=click.STRING)
@click.option("-nside", type=click.INT, help="nside for optional ud_grade.")
@click.option("-auto", is_flag=True, help="Automatically sets all plotting parameters.")
@click.option(
    "-min", default=False, help="Min value of colorbar, overrides autodetector."
)
@click.option(
    "-max", default=False, help="Max value of colorbar, overrides autodetector."
)
@click.option(
    "-minmax",
    is_flag=True,
    help="Toggle min max values to be min and max of data (As opposed to 97.5 percentile).",
)
@click.option(
    "-range",
    "rng",
    default=None,
    type=click.STRING,
    help='Color range. "-range auto" sets to 97.5 percentile of data.',
)  # str until changed to float
@click.option("-colorbar", "-bar", is_flag=True, help='Adds colorbar ("cb" in filename)')
@click.option(
    "-lmax",
    default=None,
    type=click.FLOAT,
    help="This is automatically set from the h5 file. Only available for alm inputs.",
)
@click.option(
    "-fwhm",
    default=0.0,
    type=click.FLOAT,
    help="FWHM of smoothing to apply to alm binning. Only available for alm inputs.",
)
@click.option(
    "-mask",
    default=None,
    type=click.STRING,
    help="Masks input with specified maskfile.",
)
@click.option(
    "-mfill",
    default=None,
    type=click.STRING,
    help='Color to fill masked area. for example "gray". Transparent by default.',
)
@click.option(
    "-sig",
    default=[0,],
    type=click.INT,
    multiple=True,
    help='Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)',
)
@click.option(
    "-remove_dipole",
    default=None,
    type=click.STRING,
    help="Fits a dipole to the map and removes it.",
)
@click.option(
    "-log/-no-log",
    "logscale",
    default=None,
    help="Plots using planck semi-logscale. (Autodetector sometimes uses this.)",
)
@click.option(
    "-size",
    default="m",
    type=click.STRING,
    help="Size: 1/3, 1/2 and full page width. 8.8, 12.0, 18. cm (s, m or l [small, medium or large], m by default)",
)
@click.option(
    "-white_background",
    is_flag=True,
    help="Sets the background to be white. (Transparent by default [recommended])",
)
@click.option(
    "-darkmode",
    is_flag=True,
    help='Plots all outlines in white for dark bakgrounds ("dark" in filename)',
)
@click.option("-pdf", is_flag=True, help="Saves output as .pdf ().png by default)")
@click.option(
    "-cmap",
    default=None,
    help="Choose different color map (string), such as Jet or planck",
)
@click.option(
    "-title",
    default=None,
    type=click.STRING,
    help="Set title (Upper right), has LaTeX functionality. Ex. $A_{s}$.",
)
@click.option(
    "-unit",
    default=None,
    type=click.STRING,
    help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",
)
@click.option("-verbose", is_flag=True, help="Verbose mode")
def plot(
    input,
    nside,
    auto,
    min,
    max,
    minmax,
    rng,
    colorbar,
    lmax,
    fwhm,
    mask,
    mfill,
    sig,
    remove_dipole,
    logscale,
    size,
    white_background,
    darkmode,
    pdf,
    cmap,
    title,
    unit,
    verbose,
):
    """
    \b
    Plots map from .fits.

    Uses 97.5 percentile values for min and max by default!\n
    RECCOMENDED: Use -auto to autodetect map type and set parameters.\n
    Some autodetected maps use logscale, you will be warned.
    """
    dataset = None

    from c3postproc.plotter import Plotter

    Plotter(
        input,
        dataset,
        nside,
        auto,
        min,
        max,
        minmax,
        rng,
        colorbar,
        lmax,
        fwhm,
        mask,
        mfill,
        sig,
        remove_dipole,
        logscale,
        size,
        white_background,
        darkmode,
        pdf,
        cmap,
        title,
        unit,
        verbose,
    )


@commands.command()
@click.argument("input", nargs=1, type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("nside", type=click.INT)
@click.option("-auto", is_flag=True, help="Automatically sets all plotting parameters.")
@click.option(
    "-min", default=False, help="Min value of colorbar, overrides autodetector."
)
@click.option(
    "-max", default=False, help="Max value of colorbar, overrides autodetector."
)
@click.option(
    "-minmax",
    is_flag=True,
    help="Toggle min max values to be min and max of data (As opposed to 97.5 percentile).",
)
@click.option(
    "-range",
    "rng",
    default=None,
    type=click.STRING,
    help='Color range. "-range auto" sets to 97.5 percentile of data.',
)  # str until changed to float
@click.option("-colorbar","-bar", is_flag=True, help='Adds colorbar ("cb" in filename)')
@click.option(
    "-lmax",
    default=None,
    type=click.FLOAT,
    help="This is automatically set from the h5 file. Only available for alm inputs.",
)
@click.option(
    "-fwhm",
    default=0.0,
    type=click.FLOAT,
    help="FWHM of smoothing to apply to alm binning. Only available for alm inputs.",
)
@click.option(
    "-mask",
    default=None,
    type=click.STRING,
    help="Masks input with specified maskfile.",
)
@click.option(
    "-mfill",
    default=None,
    type=click.STRING,
    help='Color to fill masked area. for example "gray". Transparent by default.',
)
@click.option(
    "-sig",
    default=[0,],
    type=click.INT,
    multiple=True,
    help='Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)',
)
@click.option(
    "-remove_dipole",
    default=None,
    type=click.STRING,
    help="Fits a dipole to the map and removes it.",
)
@click.option(
    "-log/-no-log",
    "logscale",
    default=None,
    help="Plots using planck semi-logscale. (Autodetector sometimes uses this.)",
)
@click.option(
    "-size",
    default="m",
    type=click.STRING,
    help="Size: 1/3, 1/2 and full page width. 8.8, 12.0, 18. cm (s, m or l [small, medium or large], m by default)",
)
@click.option(
    "-white_background",
    is_flag=True,
    help="Sets the background to be white. (Transparent by default [recommended])",
)
@click.option(
    "-darkmode",
    is_flag=True,
    help='Plots all outlines in white for dark bakgrounds ("dark" in filename)',
)
@click.option("-pdf", is_flag=True, help="Saves output as .pdf ().png by default)")
@click.option(
    "-cmap",
    default=None,
    help="Choose different color map (string), such as Jet or planck",
)
@click.option(
    "-title",
    default=None,
    type=click.STRING,
    help="Set title (Upper right), has LaTeX functionality. Ex. $A_{s}$.",
)
@click.option(
    "-unit",
    default=None,
    type=click.STRING,
    help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",
)
@click.option("-verbose", is_flag=True, help="Verbose mode")
def ploth5(
    input,
    dataset,
    nside,
    auto,
    min,
    max,
    minmax,
    rng,
    colorbar,
    lmax,
    fwhm,
    mask,
    mfill,
    sig,
    remove_dipole,
    logscale,
    size,
    white_background,
    darkmode,
    pdf,
    cmap,
    title,
    unit,
    verbose,
):
    """
    \b
    Plots map or alms from h5 file.\n

    c3pp ploth5 [.h5] [000004/cmb/amp_map] [nside]\n
    c3pp ploth5 [.h5] [000004/cmb/amp_alm] [nside] (Optional FWHM smoothing and LMAX for alm data).\n

    Uses 97.5 percentile values for min and max by default!\n
    RECCOMENDED: Use -auto to autodetect map type and set parameters.\n
    Some autodetected maps use logscale, you will be warned.

    """

    from c3postproc.plotter import Plotter

    Plotter(
        input,
        dataset,
        nside,
        auto,
        min,
        max,
        minmax,
        rng,
        colorbar,
        lmax,
        fwhm,
        mask,
        mfill,
        sig,
        remove_dipole,
        logscale,
        size,
        white_background,
        darkmode,
        pdf,
        cmap,
        title,
        unit,
        verbose,
    )


@commands.command()
@click.argument("filename", type=click.STRING)
@click.argument("nchains", type=click.INT)
@click.argument("burnin", type=click.INT)
@click.option("-path", default="cmb/sigma_l", help="Dataset path ex. cmb/sigma_l")
@click.argument("outname", type=click.STRING)
def sigma_l2fits(filename, nchains, burnin, path, outname, save=True):
    """
    \b
    Converts c3-h5 dataset to fits suitable for c1 BR and GBR estimator analysis.\n

    ex. c3pp sigma_l2fits chains_v1/chain 5 10 cmb_sigma_l_GBRlike.fits \n

    If "chain_c0001.h5", filename is cut to "chain" and will look in same directory for "chain_c*****.h5".\n
    See comm_like_tools for further information about BR and GBR post processing
    """
    import h5py

    if filename.endswith(".h5"):
        filename = filename.rsplit("_", 1)[0]

    temp = np.zeros(nchains)
    for nc in range(1, nchains + 1):
        with h5py.File(filename + "_c" + str(nc).zfill(4) + ".h5", "r") as f:
            groups = list(f.keys())
            temp[nc - 1] = len(groups)
    nsamples_max = int(max(temp[:]))
    click.echo("maximum number of samples for chain: " + str(nsamples_max))

    for nc in range(1, nchains + 1):
        with h5py.File(filename + "_c" + str(nc).zfill(4) + ".h5", "r") as f:
            click.echo("Reading HDF5 file: " + filename + " ...")
            groups = list(f.keys())
            click.echo()
            click.echo("Reading " + str(len(groups)) + " samples from file.")

            if nc == 1:
                dset = np.zeros(
                    (
                        nsamples_max + 1,
                        1,
                        len(f[groups[0] + "/" + path]),
                        len(f[groups[0] + "/" + path][0]),
                    )
                )
                nspec = len(f[groups[0] + "/" + path])
                lmax = len(f[groups[0] + "/" + path][0]) - 1
                nsamples = len(groups)
            else:
                nsamples = len(groups)
                dset = np.append(
                    dset, np.zeros((nsamples_max + 1, 1, nspec, lmax + 1)), axis=1
                )
            click.echo(np.shape(dset))

            click.echo(
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
                    dset[i + 1, nc - 1, j, :] = np.asarray(
                        f[groups[i] + "/" + path][j][:]
                    )
    # Optimize with jit?
    ell = np.arange(lmax + 1)
    for nc in range(1, nchains + 1):
        for i in range(1, nsamples_max + 1):
            for j in range(nspec):
                dset[i, nc - 1, j, :] = (
                    dset[i, nc - 1, j, :] * ell[:] * (ell[:] + 1.0) / 2.0 / np.pi
                )
    dset[0, :, :, :] = nsamples - burnin

    if save:
        import fitsio

        click.echo(f"Dumping fits file: {outname}...")
        dset = np.asarray(dset, dtype="f4")
        fits = fitsio.FITS(outname, mode="rw", clobber=True, verbose=True)
        h_dict = [
            {
                "name": "FUNCNAME",
                "value": "Gibbs sampled power spectra",
                "comment": "Full function name",
            },
            {"name": "LMAX", "value": lmax, "comment": "Maximum multipole moment"},
            {"name": "NUMSAMP", "value": nsamples_max, "comment": "Number of samples"},
            {
                "name": "NUMCHAIN",
                "value": nchains,
                "comment": "Number of independent chains",
            },
            {"name": "NUMSPEC", "value": nspec, "comment": "Number of power spectra"},
        ]
        fits.write(dset[:, :, :, :], header=h_dict, clobber=True)
        fits.close()
    return dset


@commands.command()
@click.argument("filename", type=click.STRING)
@click.argument("min", type=click.INT)
@click.argument("max", type=click.INT)
@click.argument("binfile", type=click.STRING)
def dlbin2dat(filename, min, max, binfile):
    """
    Outputs a .dat file of binned powerspectra averaged over a range of output samples with tilename Dl_[signal]_binned.dat.
    """
    signal = "cmb/Dl"

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

    header = f"{'l':22} {'lmin':24} {'lmax':24} {'Dl':24} {'stddev':24}"
    for signal in binned_data.keys():
        np.savetxt("Dl_" + signal + "_binned.dat", binned_data[signal], header=header)


@commands.command()
@click.argument("filename", type=click.STRING)
@click.argument("dataset", type=click.STRING)
def h5map2fits(filename, dataset, save=True):
    """
    Outputs a .h5 map to fits on the form 000001_cmb_amp_n1024.fits
    """
    import healpy as hp
    import h5py

    with h5py.File(filename, "r") as f:
        maps = f[dataset][()]
        lmax = f[dataset[:-4] + "_lmax"][()]  # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_map", "")
    if save:
        hp.write_map(outfile + f"_n{str(nside)}.fits", maps, overwrite=True)
    return maps, nside, lmax, outfile


@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("nside", type=click.INT)
@click.option("-lmax", default=None)
@click.option("-fwhm", default=0.0, type=click.FLOAT)
def alm2fits(input, dataset, nside, lmax, fwhm):
    """
    Converts c3 alms in .h5 file to fits with given nside and optional smoothing.
    """
    alm2fits_tool(input, dataset, nside, lmax, fwhm)


@commands.command()
@click.argument("chain", type=click.STRING)
@click.argument("burnin1", type=click.INT)
@click.argument("chain_resamp", type=click.STRING)
@click.argument("burnin2", type=click.INT)
@click.argument("chain_resamp_nocls", type=click.STRING)
@click.argument("procver", type=click.STRING)
@click.option("-skipcopy", is_flag=True, help="Don't copy .h5 files")
@click.option("-skipfreqmaps", is_flag=True, help="Don't output freqmaps")
@click.pass_context
def release(
    ctx,
    chain,
    burnin1,
    chain_resamp,
    burnin2,
    chain_resamp_nocls,
    procver,
    skipcopy,
    skipfreqmaps,
):
    """
    Creates a release file-set on the BeyondPlanck format.\n
    https://gitlab.com/BeyondPlanck/repo/-/wikis/BeyondPlanck-Release-Candidate-2\n
    

    ex. c3pp release chains_hm1/chain_c0001.h5 30 chains_resamp/chain_c0001.h5 5 chains_resamp_nocl/chain_c0001.h5 bpr3 \n
    Where there are different burnin variables for each of the two used hdf files. \n

    This function outputs the following files to the procver directory:\n
    BP_chain01_full_rc2.00.h5\n
    BP_resamp_chain01_full_Cl_rc2.00.h5\n
    BP_resamp_chain01_full_noCl_rc2.00.h5\n
    BP_param_full_v1.txt\n
    BP_param_resamp_Cl_v1.txt\n
    BP_param_resamp_noCl_v1.txt\n

    BP_030_IQU_full_n0512_rc2.00.fits\n
    BP_044_IQU_full_n0512_rc2.00.fits\n
    BP_070_IQU_full_n1024_rc2.00.fits\n

    BP_cmb_IQU_full_n1024_rc2.00.fits\n
    BP_synch_IQU_full_n1024_rc2.00.fits\n
    BP_freefree_I_full_n1024_rc2.00.fits\n
    BP_ame_I_full_n1024_rc2.00.fits\n

    BP_cmb_GBRlike_rc2.00.fits
    """
    # TODO
    # Use PIXWIN = True Is this always true?
    # When handling h5. Smooth in alms ? Not convert then smooth.
    # Smooth alm data to PLA standards
    # Use proper masks for output of CMB component
    # Use inpainted data as well in CMB component

    from c3postproc.fitsformatter import format_fits, get_data, get_header
    from pathlib import Path
    import shutil

    # Make procver directory if not exists
    print("{:#^80}".format("")) 
    print(f"Creating directory {procver}")
    Path(procver).mkdir(parents=True, exist_ok=True)

    """
    Copying chains files
    """
    if not skipcopy:
        # Full-mission Gibbs chain file
        print(f"Copying {chain} to {procver}/BP_resamp_chain01_full_noCl_rc2.00.h5")
        shutil.copyfile(chain, f"{procver}/BP_chain01_full_rc2.00.h5")

        # Resampled CMB-only full-mission Gibbs chain file with Cls (for BR estimator)
        print(
            f"Copying {chain_resamp} to {procver}/BP_resamp_chain01_full_noCl_rc2.00.h5"
        )
        shutil.copyfile(chain_resamp, f"{procver}/BP_resamp_chain01_full_Cl_rc2.00.h5")

        # Resampled CMB-only full-mission Gibbs chain file without Cls (for brute-force likelihood)
        print(
            f"Copying {chain_resamp_nocls} to {procver}/BP_resamp_chain01_full_noCl_rc2.00.h5"
        )
        shutil.copyfile(
            chain_resamp_nocls, f"{procver}/BP_resamp_chain01_full_noCl_rc2.00.h5"
        )

    """
    Copying parameter files
    """

    # Commander3 parameter file for main chain
    path = os.path.split(chain)[0]
    for file in os.listdir(path):
        if file.startswith("param"):
            print(f"Copying {path}/{file} to {procver}/BP_param_full_v1.txt")
            shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_full_v1.txt")

    # Commander3 parameter file for CMB resampling chain with Cls (for BR)
    path = os.path.split(chain_resamp)[0]
    for file in os.listdir(path):
        if file.startswith("param"):
            print(f"Copying {path}/{file} to {procver}/BP_param_resamp_Cl_v1.txt")
            shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_resamp_Cl_v1.txt")

    # Commander3 parameter file for CMB resampling chain without Cls (for brute-force likelihood)
    path = os.path.split(chain_resamp_nocls)[0]
    for file in os.listdir(path):
        if file.startswith("param"):
            print(f"Copying {path}/{file} to {procver}/BP_param_resamp_noCl_v1.txt")
            shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_resamp_noCl_v1.txt")

    """
    IQU mean, IQU stdev, (Masks for cmb)
    Run mean and stddev from min to max sample (Choose min manually or start at 1?)
    """
    if not skipfreqmaps:
        # Full-mission 30 GHz IQU frequency map
        # BP_030_IQU_full_n0512_rc2.00.fits
        format_fits(
            chain=chain,
            extname="FREQMAP",
            types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS",],
            units=["uK", "uK", "uK", "uK", "uK", "uK",],
            nside=512,
            burnin=burnin1,
            polar=True,
            component="030",
            fwhm=0.0,
            nu_ref_t="30.0 GHz",
            nu_ref_p="30.0 GHz",
            procver=procver,
            filename="BP_030_IQU_full_n0512_rc2.00.fits",
            bndctr=30,
            restfreq=28.456,
            bndwid=9.899,
        )
        # Full-mission 44 GHz IQU frequency map
        format_fits(
            chain=chain,
            extname="FREQMAP",
            types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS",],
            units=["uK", "uK", "uK", "uK", "uK", "uK",],
            nside=512,
            burnin=burnin1,
            polar=True,
            component="030",
            fwhm=0.0,
            nu_ref_t="44.0 GHz",
            nu_ref_p="44.0 GHz",
            procver=procver,
            filename="BP_044_IQU_full_n0512_rc2.00.fits",
            bndctr=44,
            restfreq=44.121,
            bndwid=10.719,
        )
        # Full-mission 70 GHz IQU frequency map
        format_fits(
            chain=chain,
            extname="FREQMAP",
            types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS",],
            units=["uK", "uK", "uK", "uK", "uK", "uK",],
            nside=1024,
            burnin=burnin1,
            polar=True,
            component="070",
            fwhm=0.0,
            nu_ref_t="70.0 GHz",
            nu_ref_p="70.0 GHz",
            procver=procver,
            filename="BP_070_IQU_full_n1024_rc2.00.fits",
            bndctr=70,
            restfreq=70.467,
            bndwid=14.909,
        )

    """
    FOREGROUND MAPS
    """
    # Full-mission CMB IQU map
    format_fits(
        chain_resamp,
        extname="COMP-MAP-CMB",
        types=[
            "I_MEAN",
            "Q_MEAN",
            "U_MEAN",
            "I_RMS",
            "Q_RMS",
            "U_RMS",
            "mask1",
            "mask2",
        ],
        units=["uK_cmb", "uK_cmb", "uK_cmb", "uK", "uK", "uK", "NONE", "NONE",],
        nside=1024,
        burnin=burnin2,
        polar=True,
        component="CMB",
        fwhm=0.0,
        nu_ref_t="NONE",  # TODO What is this?
        nu_ref_p="NONE",
        procver=procver,
        filename="BP_cmb_IQU_full_n1024_rc2.00.fits",
        bndctr=None,
        restfreq=None,
        bndwid=None,
    )
    # Full-mission synchrotron IQU map
    format_fits(
        chain,
        extname="COMP-MAP-SYNCHROTRON",
        types=[
            "I_MEAN",
            "Q_MEAN",
            "U_MEAN",
            "BETA_MEAN",
            "BETA_P_MEAN",
            "I_RMS",
            "Q_RMS",
            "U_RMS",
            "BETA_RMS",
            "BETA_P_RMS",
        ],
        units=[
            "uK_RJ",
            "uK_RJ",
            "uK_RJ",
            "NONE",
            "NONE",
            "uK",
            "uK",
            "uK",
            "NONE",
            "NONE",
        ],
        nside=1024,
        burnin=burnin1,
        polar=True,
        component="SYNCHROTRON",
        fwhm=60.0,
        nu_ref_t="0.408 GHz",
        nu_ref_p="30.0 GHz",
        procver=procver,
        filename="BP_synch_IQU_full_n1024_rc2.00.fits",
        bndctr=None,
        restfreq=None,
        bndwid=None,
    )
    # Full-mission free-free I map
    format_fits(
        chain,
        extname="COMP-MAP-FREE-FREE",
        types=["I_MEAN", "TE_MEAN", "I_RMS", "TE_RMS",],
        units=["uK_RJ", "K", "uK_RJ", "K",],
        nside=1024,
        burnin=burnin1,
        polar=False,
        component="FREE-FREE",
        fwhm=75.0,
        nu_ref_t="40.0 GHz",
        nu_ref_p="40.0 GHz",
        procver=procver,
        filename="BP_freefree_I_full_n1024_rc2.00.fits",
        bndctr=None,
        restfreq=None,
        bndwid=None,
    )

    # Full-mission AME I map
    format_fits(
        chain,
        extname="COMP-MAP-AME",
        types=["I_MEAN", "NU_P_MEAN", "I_RMS", "NU_P_RMS"],
        units=["uK_RJ", "GHz", "uK_RJ", "GHz",],
        nside=1024,
        burnin=burnin1,
        polar=False,
        component="AME",
        fwhm=90.0,
        nu_ref_t="22.0 GHz",
        nu_ref_p="22.0 GHz",
        procver=procver,
        filename="BP_ame_I_full_n1024_rc2.00.fits",
        bndctr=None,
        restfreq=None,
        bndwid=None,
    )
    """ As implemented by Simone
    """
    # Gaussianized TT Blackwell-Rao input file
    print()
    print("{:-^50}".format("CMB GBR")) 
    ctx.invoke(
        sigma_l2fits,
        filename=chain_resamp,
        nchains=1,
        burnin=burnin2,
        path="cmb/sigma_l",
        outname=f"{procver}/BP_cmb_GBRlike_rc2.00.fits",
        save=True,
    )

    """
    TODO Generalize this so that they can be generated by Elina and Anna-Stiina
    """
    # Full-mission 30 GHz IQU beam symmetrized frequency map
    # BP_030_IQUdeconv_full_n0512_rc2.00.fits
    # Full-mission 44 GHz IQU beam symmetrized frequency map
    # BP_044_IQUdeconv_full_n0512_rc2.00.fits
    # Full-mission 70 GHz IQU beam symmetrized frequency map
    # BP_070_IQUdeconv_full_n1024_rc2.00.fits

    """ Both sigma_l's and Dl's re in the h5. (Which one do we use?)
    """
    # CMB TT, TE, EE power spectrum
    # BP_cmb_Cl_rc2.00.txt

    """ Just get this from somewhere
    """
    # Best-fit LCDM CMB TT, TE, EE power spectrum
    # BP_cmb_bfLCDM_rc2.00.txt
