import time
import os
import numpy as np
import sys
import click
from c3postproc.tools import *


@click.group()
def commands():
    pass

"""
FITS COMMANDS GO HERE
"""

@commands.command()
@click.argument("input", type=click.STRING)
def printheader(input,):
    """
    Prints the header of a fits file.
    """
    from astropy.io import fits

    with fits.open(input) as hdulist:
        hdulist.info()
        for hdu in hdulist:
            print(repr(hdu.header))

@commands.command()
@click.argument("input", type=click.STRING)
def printdata(input,):
    """
    Prints the data of a fits file
    """
    from astropy.io import fits

    with fits.open(input) as hdulist:
        hdulist.info()
        for hdu in hdulist:
            print(repr(hdu.data))


@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.argument("columns", type=click.INT)
def rmcolumn(input, output, columns):
    """
    remove columns in fits file
    """
    from astropy.io import fits

    with fits.open(input) as hdulist:
        for hdu in hdulist:
            hdu.header.pop(columns)        
            hdu.data.del_col(columns)        
        hdulist.writeto(output, overwrite=True)
        

@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
def QU2ang(input, output,):
    """
    remove columns in fits file
    """
    import healpy as hp
    Q, U = hp.read_map(input, field=(1,2), dtype=None, verbose=False)
    phi = 0.5*np.arctan(U,Q)
    hp.write_map(output, phi, dtype=None, overwrite=True)





@commands.command()
@click.argument("input1", type=click.STRING)
@click.argument("input2", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-beam1", type=click.STRING, help="Optional beam file for input 1",)
@click.option("-beam2", type=click.STRING, help="Optional beam file for input 2",)
@click.option("-mask", type=click.STRING, help="Mask",)
def crosspec(input1, input2, output, beam1, beam2, mask,):
    """
    This function calculates a powerspectrum from polspice using this path:
    /mn/stornext/u3/trygvels/PolSpice_v03-03-02/
    """
    sys.path.append("/mn/stornext/u3/trygvels/PolSpice_v03-03-02/")
    from ispice import ispice

    lmax = 6000
    fwhm = 0
    #mask = "dx12_v3_common_mask_pol_005a_2048_v2.fits"
    if beam1 and beam2:
        ispice(input1,
               clout=output,
               nlmax=lmax,
               beam_file1=beam1,
               beam_file2=beam2,
               mapfile2=input2,
               maskfile1=mask,
               maskfile2=mask,
               fits_out="NO",
               polarization="YES",
               subav="YES",
               subdipole="YES",
               symmetric_cl="YES",
           )
    else:

        ispice(input1,
               clout=output,
               nlmax=lmax,
               beam1=0.0,
               beam2=0.0,
               mapfile2=input2,
               maskfile1=mask,
               maskfile2=mask,
               fits_out="NO",
               polarization="YES",
               subav="YES",
               subdipole="YES",
               symmetric_cl="YES",
           )

@commands.command()
@click.argument("input", type=click.STRING)
def specplot(input,):
    """
    This function plots the file output by the Crosspec function.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.2})
    sns.set_style("whitegrid")
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    }
    sns.set_style(custom_style)

    lmax = 350
    ell, ee, bb, eb = np.loadtxt(input, usecols=(0,2,3,6), skiprows=3, max_rows=lmax, unpack=True)
                                 
    ee = ee*(ell*(ell+1)/(2*np.pi))
    bb = bb*(ell*(ell+1)/(2*np.pi))
    eb = eb*(ell*(ell+1)/(2*np.pi))
	
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.loglog(ell, ee, linewidth=2, label="EE")
    ax1.loglog(ell, bb, linewidth=2, label="BB")
    ax1.set_ylabel(r"$D_l$ [$\mu K^2$]")

    ax2.semilogx(ell, bb/ee, linewidth=2, label="BB/EE")
    ax2.set_ylabel(r"BB/EE")
    ax2.set_xlabel(r'Multipole moment, $l$')
    #plt.semilogx(ell, eb, label="EB")
	
    ax1.axhline(y=0, color="black", linestyle='--', zorder=5, linewidth=0.5)
    sns.despine(top=True, right=True, left=False, bottom=False, ax=ax1)
    sns.despine(top=True, right=True, left=False, bottom=False, ax=ax2)
    #plt.xlim(0,lmax)
    ax1.set_ylim(0.1,150)
    ax2.set_ylim(-1,2.5)
    #ax.axes.xaxis.grid()
    ax1.legend(frameon=False)

    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.savefig(input.replace(".dat",".pdf"), dpi=300)
    plt.show()

@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for alm binning",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def mean(input, dataset, output, min, max, maxchain, fwhm, nside, zerospin, pixweight):
    """
    Calculates the mean over sample range from .h5 file.\n
    ex. chains_c0001.h5 dust/amp_map 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -maxchain 3\n
    ex. chains_c0001.h5 dust/amp_alm 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -nside 512\n
    If output name is set to .dat, data will not be converted to map.
    """
    if dataset.endswith("alm") and nside == None:
        click.echo("Please specify nside when handling alms.")
        sys.exit()


    h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, np.mean, pixweight, zerospin,)
    #h5handler_old(input, dataset, min, max, maxchain, output, fwhm, nside, np.mean)

@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for alm binning",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def stddev(input, dataset, output, min, max, maxchain, fwhm, nside, zerospin, pixweight,):
    """
    Calculates the stddev over sample range from .h5 file.\n
    ex. chains_c0001.h5 dust/amp_map 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -maxchain 3\n
    ex. chains_c0001.h5 dust/amp_alm 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -nside 512\n

    If output name is set to .dat, data will not be converted to map.
    """
    if dataset.endswith("alm") and nside == None:
        click.echo("Please specify nside when handling alms.")
        sys.exit()

    h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, np.std, pixweight, zerospin,)
    #h5handler_old(input, dataset, min, max, maxchain, output, fwhm, nside, np.std,)

@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for down-grading maps before calculation",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-missing", is_flag=True, help="If files are missing, drop them. Else, exit computation",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def fits_mean(
        input, output, min, max, maxchain, fwhm, nside, zerospin, missing, pixweight):
    """
    Calculates the mean over sample range from fits-files.\n
    ex. res_030_c0001_k000020.fits res_030_20-100_mean_40arcmin.fits -min 20 -max 100 -fwhm 40 -maxchain 3\n
    If output name is set to .dat, data will not be converted to map.
    """

    fits_handler(input, min, max, maxchain, output, fwhm, nside, zerospin, missing, pixweight, False, np.mean)

@commands.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for down-grading maps before calculation",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-missing", is_flag=True, help="If files are missing, drop them. Else, exit computation",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def fits_stddev(
        input, output, min, max, maxchain, fwhm, nside, zerospin, missing, pixweight):
    """
    Calculates the standard deviation over sample range from fits-files.\n
    ex. res_030_c0001_k000020.fits res_030_20-100_mean_40arcmin.fits -min 20 -max 100 -fwhm 40 -maxchain 3\n
    If output name is set to .dat, data will not be converted to map.
    """

    fits_handler(input, min, max, maxchain, output, fwhm, nside, zerospin, missing, pixweight, False, np.std)

@commands.command()
@click.argument("input", type=click.Path(exists=True))#, nargs=-1,)
@click.option("-dataset", type=click.STRING, help="for .h5 plotting (ex. 000007/cmb/amp_alm)")
@click.option("-nside", type=click.INT, help="nside for optional ud_grade.",)
@click.option("-auto", is_flag=True, help="Automatically sets all plotting parameters.",)
@click.option("-min", default=False, help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", default=False, help="Max value of colorbar, overrides autodetector.",)
@click.option("-mid", default=0.0, type=click.FLOAT, multiple=True, help='Adds tick values "-mid 2 -mid 4"',)
@click.option("-range", default="auto", type=click.STRING, help='Color range. "-range auto" sets to 97.5 percentile of data., or "minmax" which sets to data min and max values.',)  # str until changed to float
@click.option("-colorbar", "-bar", is_flag=True, help='Adds colorbar ("cb" in filename)',)
@click.option("-lmax", default=None, type=click.FLOAT, help="This is automatically set from the h5 file. Only available for alm inputs.",)
@click.option("-fwhm", default=0.0, type=click.FLOAT, help="FWHM of smoothing.",)
@click.option("-mask", default=None, type=click.STRING, help="Masks input with specified maskfile.",)
@click.option("-mfill", default=None, type=click.STRING, help='Color to fill masked area. for example "gray". Transparent by default.',)
@click.option("-sig", default=[0,], type=click.INT, multiple=True, help="Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)",)
@click.option("-remove_dipole", default=None, type=click.STRING, help="Fits a dipole to the map and removes it.",)
@click.option("-log/-no-log", "logscale", default=None, help="Plots using planck semi-logscale. (Autodetector sometimes uses this.)",)
@click.option("-size", default="m", type=click.STRING, help="Size: 1/3, 1/2 and full page width (8.8/12/18cm) [ex. s, m or l, or ex. slm for all], m by default",)
@click.option("-white_background", is_flag=True, help="Sets the background to be white. (Transparent by default [recommended])",)
@click.option("-darkmode", is_flag=True, help='Plots all outlines in white for dark bakgrounds ("dark" in filename)',)
@click.option("-png", is_flag=True, help="Saves output as .png ().pdf by default)",)
@click.option("-cmap", default=None, help="Choose different color map (string), such as Jet or planck",)
@click.option("-title", default=None, type=click.STRING, help="Set title (Upper right), has LaTeX functionality. Ex. $A_{s}$.",)
@click.option("-ltitle", default=None, type=click.STRING, help="Set title (Upper left), has LaTeX functionality. Ex. $A_{s}$.",)
@click.option("-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-scale", default=1.0, type=click.FLOAT, help="Scale input map [ex. 1e-6 for muK to K]",)
@click.option("-outdir", type=click.Path(exists=True), help="Output directory for plot",)
@click.option("-verbose", is_flag=True, help="Verbose mode")
def plot(input, dataset, nside, auto, min, max, mid, range, colorbar, lmax, fwhm, mask, mfill, sig, remove_dipole, logscale, size, white_background, darkmode, png, cmap, title, ltitle, unit, scale, outdir, verbose,):
    """
    \b
    Plots map from .fits or h5 file.
    ex. c3pp plot coolmap.fits -bar -auto -lmax 60 -darkmode -pdf -title $\beta_s$
    ex. c3pp plot coolhdf.h5 -dataset 000007/cmb/amp_alm -nside 512 -remove_dipole maskfile.fits -cmap cmasher.arctic 

    Uses 97.5 percentile values for min and max by default!\n
    RECCOMENDED: Use -auto to autodetect map type and set parameters.\n
    Some autodetected maps use logscale, you will be warned.
    """
    if input.endswith(".h5") and not dataset and not nside:
        print("Specify Nside when plotting alms!")
        sys.exit()
        
    data = None # Only used if calling plotter directly for plotting data array
    from c3postproc.plotter import Plotter
    Plotter(input, dataset, nside, auto, min, max, mid, range, colorbar, lmax, fwhm, mask, mfill, sig, remove_dipole, logscale, size, white_background, darkmode, png, cmap, title, ltitle, unit, scale, outdir, verbose,data)


@commands.command()
@click.argument("filename", type=click.STRING)
@click.argument("nchains", type=click.INT)
@click.argument("burnin", type=click.INT)
@click.option("-path", default="cmb/sigma_l", help="Dataset path ex. cmb/sigma_l",)
@click.argument("outname", type=click.STRING)
def sigma_l2fits(filename, nchains, burnin, path, outname, save=True,):
    """
    \b
    Converts c3-h5 dataset to fits suitable for c1 BR and GBR estimator analysis.\n

    ex. c3pp sigma-l2fits chains_v1/chain 5 10 cmb_sigma_l_GBRlike.fits \n

    If "chain_c0001.h5", filename is cut to "chain" and will look in same directory for "chain_c*****.h5".\n
    See comm_like_tools for further information about BR and GBR post processing
    """
    #data = h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, np.mean, pixweight, zerospin,)
    print("{:-^48}".format("Formatting sigma_l data to fits file"))
    import h5py

    if filename.endswith(".h5"):
        filename = filename.rsplit("_", 1)[0]

    temp = np.zeros(nchains)
    for nc in range(1, nchains + 1):
        with h5py.File(filename + "_c" + str(nc).zfill(4) + ".h5", "r",) as f:
            groups = list(f.keys())
            temp[nc - 1] = len(groups)
    nsamples_max = int(max(temp[:]))
    print(f"Largest chain has {nsamples_max} samples, using burnin {burnin}\n")

    for nc in range(1, nchains + 1):
        fn = filename + "_c" + str(nc).zfill(4) + ".h5"
        with h5py.File(fn, "r",) as f:
            print(f"Reading {fn}")
            groups = list(f.keys())
            nsamples = len(groups)
            if nc == 1:
                dset = np.zeros((nsamples_max + 1, 1, len(f[groups[0] + "/" + path]), len(f[groups[0] + "/" + path][0]),))
                nspec = len(f[groups[0] + "/" + path])
                lmax = len(f[groups[0] + "/" + path][0]) - 1
            else:
                dset = np.append(dset, np.zeros((nsamples_max + 1, 1, nspec, lmax + 1,)), axis=1,)
            print(f"Dataset: {path} \n# samples: {nsamples} \n# spectra: {nspec} \nlmax: {lmax}")

            for i in range(nsamples):
                for j in range(nspec):
                    dset[i + 1, nc - 1, j, :] = np.asarray(f[groups[i] + "/" + path][j][:])

            print("")

    # Optimize with jit?
    ell = np.arange(lmax + 1)
    for nc in range(1, nchains + 1):
        for i in range(1, nsamples_max + 1):
            for j in range(nspec):
                dset[i, nc - 1, j, :] = dset[i, nc - 1, j, :] * ell[:] * (ell[:] + 1.0) / 2.0 / np.pi
    dset[0, :, :, :] = nsamples - burnin

    if save:
        print(f"Dumping fits file: {outname}")
        dset = np.asarray(dset, dtype="f4")

        from astropy.io import fits
        head = fits.Header()
        head["FUNCNAME"] = ("Gibbs sampled power spectra",  "Full function name")
        head["LMAX"]     = (lmax,  "Maximum multipole moment")
        head["NUMSAMP"]  = (nsamples_max,  "Number of samples")
        head["NUMCHAIN"] = (nchains,  "Number of independent chains")
        head["NUMSPEC"]  = (nspec,  "Number of power spectra")
        fits.writeto(outname, dset, head, overwrite=True)                

        # FITSIO Saving Deprecated (Use astropy)
        if False:
            import fitsio

            fits = fitsio.FITS(outname, mode="rw", clobber=True, verbose=True,)
            h_dict = [
                {"name": "FUNCNAME", "value": "Gibbs sampled power spectra", "comment": "Full function name",},
                {"name": "LMAX", "value": lmax, "comment": "Maximum multipole moment",},
                {"name": "NUMSAMP", "value": nsamples_max, "comment": "Number of samples",},
                {"name": "NUMCHAIN", "value": nchains, "comment": "Number of independent chains",},
                {"name": "NUMSPEC", "value": nspec, "comment": "Number of power spectra",},
            ]
            fits.write(dset[:, :, :, :], header=h_dict, clobber=True,)
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
    possible_signals = ["TT","EE","BB","TE","EB","TB",]
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
            binned_data[signal].append([ellcenter, lmin, lmax, np.mean(dats[:, signal_id, lmin], axis=0,), np.std(dats[:, signal_id, lmin], axis=0,),])

    header = f"{'l':22} {'lmin':24} {'lmax':24} {'Dl':24} {'stddev':24}"
    for signal in binned_data.keys():
        np.savetxt("Dl_" + signal + "_binned.dat", binned_data[signal], header=header,)


# ISSUE!
# @commands.command()
# @click.argument("filename", type=click.STRING)
# @click.argument("dataset", type=click.STRING)
# @click.option("-save", is_flag=True, default=True)
def h5map2fits(filename, dataset, save=True):
    """
    Outputs a .h5 map to fits on the form 000001_cmb_amp_n1024.fits
    """
    import healpy as hp
    import h5py

    dataset, tag = dataset.rsplit("/", 1)

    with h5py.File(filename, "r") as f:
        maps = f[f"{dataset}/{tag}"][()]
        lmax = f[f"{dataset}/amp_lmax"][()]  # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    dataset = f"{dataset}/{tag}"
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_map", "")
    if save:
        hp.write_map(outfile + f"_n{str(nside)}.fits", maps, overwrite=True,)
    return maps, nside, lmax, outfile

@commands.command()
@click.argument("filename", type=click.STRING)
@click.argument("dataset", type=click.STRING)
def h52fits(filename, dataset,):
    """
    Outputs a .h5 map to fits on the form 000001_cmb_amp_n1024.fits
    """
    import healpy as hp
    import h5py

    dataset, tag = dataset.rsplit("/", 1)

    with h5py.File(filename, "r") as f:
        maps = f[f"{dataset}/{tag}"][()]
        if ('aml' in tag):
            lmax = f[f"{dataset}/amp_lmax"][()]  # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    dataset = f"{dataset}/{tag}"
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_map", "")
    hp.write_map(outfile + f"_n{str(nside)}.fits", maps, overwrite=True,)
    

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
@click.argument("procver", type=click.STRING)
@click.option("-mask", type=click.Path(exists=True), help="Mask for calculating cmb",)
@click.option("-defaultmask", is_flag=True, help="Use default dx12 mask",)
@click.option("-freqmaps", is_flag=True, help=" output freqmaps",)
@click.option("-cmb", is_flag=True, help=" output cmb",)
@click.option("-synch", is_flag=True, help=" output synch",)
@click.option("-ame", is_flag=True, help=" output ame",)
@click.option("-ff", is_flag=True, help=" output ff",)
@click.option("-dust", is_flag=True, help=" output dust",)
@click.option("-diff", is_flag=True, help="Creates diff maps to dx12 and npipe")
@click.option("-diffcmb", is_flag=True, help="Creates diff maps with cmb maps")
@click.option("-spec", is_flag=True, help="Creates emission plot")
@click.option("-all", "all_", is_flag=True, help="Output all")
@click.pass_context
def plotrelease(ctx, procver, mask, defaultmask, freqmaps, cmb, synch, ame, ff, dust, diff, diffcmb, spec, all_):
    """
    \b
    Plots all release files\n
    """
    import os
    if not os.path.exists("figs"):
        os.mkdir("figs")

    if all_:
        freqmaps = cmb = synch = ame = ff = dust = diff = diffcmb = spec = True
        defaultmask = True if not mask else False
        
    for size in ["m", "l", "s",]:
        for colorbar in [True, False]:
            if cmb and mask or defaultmask:
                outdir = "figs/cmb/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                if defaultmask:
                    mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"

                # CMB I with dip
                ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True,  range=3400)
                # CMB I no dip
                ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask, )
                ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask,  fwhm=np.sqrt(60.0**2-14**2),)
                ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask,  fwhm=np.sqrt(420.0**2-14**2),range=150)

                # CMB QU at 14 arcmin, 1 degree and 7 degree smoothing
                for hehe, fwhm in enumerate([0.0, np.sqrt(60.0**2-14**2), np.sqrt(420.0**2-14**2)]):
                    rng = 5 if hehe == 2 else None
                    ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=fwhm, range=rng)

                # RMS maps
                ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4, 5, 6,], )

            if freqmaps:
                outdir = "figs/freqmaps/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                # 030 GHz IQU
                ctx.invoke(plot, input=f"BP_030_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=3400,)
                ctx.invoke(plot, input=f"BP_030_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4,],  min=0.0, max=20.0)
                ctx.invoke(plot, input=f"BP_030_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=60.0, range=30,)
                ctx.invoke(plot, input=f"BP_030_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,],  fwhm=60.0, min=0.0, max=100,)
                ctx.invoke(plot, input=f"BP_030_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[5, 6,],  fwhm=60.0,min=0.0, max=20.0)
                ctx.invoke(plot, input=f"BP_030_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[7,],  fwhm=60.0,min=0.0, max=40.0)

                # 044 GHz IQU
                ctx.invoke(plot, input=f"BP_044_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=3400,)
                ctx.invoke(plot, input=f"BP_044_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4,],  min=0.0, max=20.0)
                ctx.invoke(plot, input=f"BP_044_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=60.0, range=30,)
                ctx.invoke(plot, input=f"BP_044_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,],  fwhm=60.0, min=0.0, max=100,)
                ctx.invoke(plot, input=f"BP_044_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[5, 6,],  fwhm=60.0,min=0.0, max=20.0)
                ctx.invoke(plot, input=f"BP_044_IQU_full_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[7,],  fwhm=60.0, min=0.0, max=40.0)
                # 070 GHz IQU
                ctx.invoke(plot, input=f"BP_070_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=3400,)
                ctx.invoke(plot, input=f"BP_070_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4,],  min=0.0, max=20.0)
                ctx.invoke(plot, input=f"BP_070_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=60.0, range=30,)
                ctx.invoke(plot, input=f"BP_070_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,],  fwhm=60.0, min=0.0, max=200,)
                ctx.invoke(plot, input=f"BP_070_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[5, 6,],  fwhm=60.0,min=0.0, max=20.0)
                ctx.invoke(plot, input=f"BP_070_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[7,],  fwhm=60.0,min=0.0, max=40.0)

            if synch:
                outdir = "figs/synchrotron/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                # Synch IQU
                ctx.invoke(plot, input=f"BP_synch_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], )

            if ff:
                outdir = "figs/freefree/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                # freefree mean and rms
                ctx.invoke(plot, input=f"BP_freefree_I_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3], )

            if ame:
                outdir = "figs/ame/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                # ame mean and rms
                ctx.invoke(plot, input=f"BP_ame_I_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3], )

            if dust:
                outdir = "figs/dust/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                # dust IQU
                ctx.invoke(plot, input=f"BP_dust_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], )

            if diff:
                outdir = "figs/freqmap_difference/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                # Plot difference to npipe and dx12
                ctx.invoke(plot, input=f"BP_030_diff_npipe_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"BP_030_diff_npipe_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"BP_030_diff_dx12_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"BP_030_diff_dx12_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)

                ctx.invoke(plot, input=f"BP_044_diff_npipe_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"BP_044_diff_npipe_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"BP_044_diff_dx12_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"BP_044_diff_dx12_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                
                ctx.invoke(plot, input=f"BP_070_diff_npipe_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"BP_070_diff_npipe_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"BP_070_diff_dx12_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"BP_070_diff_dx12_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)

            if diffcmb:
                outdir = "figs/cmb_difference/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"
                for i, method in enumerate(["Commander", "SEVEM", "NILC", "SMICA",]):
                    input = f"BP_cmb_diff_{method.lower()}_{procver}.fits"
                    ctx.invoke(plot, input=input, size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask, sig=[0,],  range=10, title=method, ltitle=" ",)
                    ctx.invoke(plot, input=input, size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4, title=method, ltitle=" ",)

            if spec: 
                print("Plotting sky model SED spectrum")
                print("Reading data")
                import healpy as hp
                maskpath="/mn/stornext/u3/trygvels/compsep/cdata/like/sky-model/masks"
                fg_path="/mn/stornext/u3/trygvels/compsep/cdata/like/sky-model/fgs_60arcmin"

                a_cmb = None

                a_s = hp.read_map(f"BP_synch_IQU_full_n1024_{procver}.fits", field=(0,1,2), dtype=None, verbose=False)
                b_s = hp.read_map(f"BP_synch_IQU_full_n1024_{procver}.fits", field=(4,5), dtype=None, verbose=False)
                
                a_ff = hp.read_map(f"BP_freefree_I_full_n1024_{procver}.fits", field=(0,), dtype=None, verbose=False)
                a_ff = hp.smoothing(a_ff, fwhm=arcmin2rad(np.sqrt(60.0**2-30**2)), verbose=False)
                t_e  = hp.read_map(f"BP_freefree_I_full_n1024_{procver}.fits", field=(1,), dtype=None, verbose=False)

                a_ame1 = hp.read_map(f"BP_ame_I_full_n1024_{procver}.fits", field=(0,), dtype=None, verbose=False)
                a_ame1 = hp.smoothing(a_ame1, fwhm=arcmin2rad(np.sqrt(60.0**2-30**2)), verbose=False)
                nup    = hp.read_map(f"BP_ame_I_full_n1024_{procver}.fits", field=(1,), dtype=None, verbose=False)                
                a_ame2 = None

                a_d = hp.read_map(f"BP_dust_IQU_full_n1024_{procver}.fits", field=(0,1,2), dtype=None, verbose=False)
                a_d = hp.smoothing(a_d, fwhm=arcmin2rad(np.sqrt(60.0**2-10**2)), verbose=False)
                b_d = hp.read_map(f"BP_dust_IQU_full_n1024_{procver}.fits", field=(4,5,), dtype=None, verbose=False)                                
                t_d = hp.read_map(f"BP_dust_IQU_full_n1024_{procver}.fits", field=(6,7,), dtype=None, verbose=False)                                

                a_co10=f"{fg_path}/co10_npipe_60arcmin.fits"
                a_co21=f"{fg_path}/co21_npipe_60arcmin.fits"
                a_co32=f"{fg_path}/co32_npipe_60arcmin.fits"
                
                mask1=f"{maskpath}/mask_70GHz_t7.fits"
                mask2=f"{maskpath}/mask_70GHz_t100.fits"
                
                print("Data read, making plots, this may take a while")
                for long in [True, False]:
                    for pol in [True, False]:
                        ctx.invoke(output_sky_model, long=long,
                                   darkmode=False, png=False,
                                   nside=64, a_cmb=a_cmb, a_s=a_s, b_s=b_s, a_ff=a_ff,
                                   t_e=t_e, a_ame1=a_ame1, a_ame2=a_ame2, nup=nup, a_d=a_d, b_d=b_d,
                                   t_d=t_d, a_co10=a_co10, a_co21=a_co21, a_co32=a_co32, mask1=mask1,
                                   mask2=mask2,)

@commands.command()
@click.argument("chain", type=click.Path(exists=True), nargs=-1,)
@click.argument("burnin", type=click.INT)
@click.argument("procver", type=click.STRING)
@click.option("-resamp", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-copy", "copy_", is_flag=True, help=" copy full .h5 file",)
@click.option("-freqmaps", is_flag=True, help=" output freqmaps",)
@click.option("-ame", is_flag=True, help=" output ame",)
@click.option("-ff", "-freefree", "ff", is_flag=True, help=" output freefree",)
@click.option("-cmb", is_flag=True, help=" output cmb",)
@click.option("-synch", is_flag=True, help=" output synchrotron",)
@click.option("-dust", is_flag=True, help=" output dust",)
@click.option("-br", is_flag=True, help=" output BR",)
@click.option("-diff", is_flag=True, help="Creates diff maps to dx12 and npipe")
@click.option("-diffcmb", is_flag=True, help="Creates diff maps cmb")
@click.option("-all", "all_", is_flag=True, help="Output all")
@click.pass_context
def release(ctx, chain, burnin, procver, resamp, copy_, freqmaps, ame, ff, cmb, synch, dust, br, diff, diffcmb, all_):
    """
    Creates a release file-set on the BeyondPlanck format.\n
    https://gitlab.com/BeyondPlanck/repo/-/wikis/BeyondPlanck-Release-Candidate-2\n    

    ex. c3pp release chains_v1_c{1,2}/chain_c000{1,2}.h5 30 BP_r1 \n
    Will output formatted files using all chains specified, \n
    with a burnin of 30 to a directory called BP_r1

    This function outputs the following files to the {procver} directory:\n
    BP_chain01_full_{procver}.h5\n
    BP_resamp_chain01_full_Cl_{procver}.h5\n
    BP_resamp_chain01_full_noCl_{procver}.h5\n
    BP_param_full_v1.txt\n
    BP_param_resamp_Cl_v1.txt\n
    BP_param_resamp_noCl_v1.txt\n

    BP_030_IQU_full_n0512_{procver}.fits\n
    BP_044_IQU_full_n0512_{procver}.fits\n
    BP_070_IQU_full_n1024_{procver}.fits\n

    BP_cmb_IQU_full_n1024_{procver}.fits\n
    BP_synch_IQU_full_n1024_{procver}.fits\n
    BP_freefree_I_full_n1024_{procver}.fits\n
    BP_ame_I_full_n1024_{procver}.fits\n

    BP_cmb_GBRlike_{procver}.fits
    """
    # TODO
    # Use proper masks for output of CMB component
    # Use inpainted data as well in CMB component

    from c3postproc.fitsformatter import format_fits, get_data, get_header
    from pathlib import Path
    import shutil

    if all_: # sets all other flags to true
        copy_ = freqmaps = ame = ff = cmb = synch = dust = br = diff = diffcmb = True

    # Make procver directory if not exists
    print("{:#^80}".format(""))
    print(f"Creating directory {procver}")
    Path(procver).mkdir(parents=True, exist_ok=True)
    chains = chain
    maxchain = len(chains)

    """
    Copying chains files
    """
    if copy_:
        # Commander3 parameter file for main chain
        for i, chainfile in enumerate(chains, 1):
            path = os.path.split(chainfile)[0]
            for file in os.listdir(path):
                if file.startswith("param") and i == 1:  # Copy only first
                    print(f"Copying {path}/{file} to {procver}/BP_param_full_c" + str(i).zfill(4) + ".txt")
                    shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_full_c" + str(i).zfill(4) + ".txt",)

            # Full-mission Gibbs chain file
            print(f"Copying {chainfile} to {procver}/BP_c" + str(i).zfill(4) + f"_full_{procver}.h5")
            shutil.copyfile(chainfile, f"{procver}/BP_c" + str(i).zfill(4) + f"_full_{procver}.h5",)

     #if halfring:
     #   # Copy halfring files
     #   for i, chainfile in enumerate([halfring], 1):
     #       # Copy halfring files
     #       print(f"Copying {resamp} to {procver}/BP_halfring_c" + str(i).zfill(4) + f"_full_Cl_{procver}.h5")
     #       shutil.copyfile(halfring, f"{procver}/BP_halfring_c" + str(i).zfill(4) + f"_full_Cl_{procver}.h5",)
        

    if resamp:
        # Commander3 parameter file for main chain
        for i, chainfile in enumerate([resamp], 1):
            # Commander3 parameter file for CMB resampling chain with Cls (for BR)
            path = os.path.split(chainfile)[0]
            for file in os.listdir(path):
                if file.startswith("param") and i == 0:
                    print(f"Copying {path}/{file} to {procver}/BP_param_resamp_Cl_c" + str(i).zfill(4) + ".txt")
                    shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_resamp_Cl_c" + str(i).zfill(4) + ".txt",)

            # Resampled CMB-only full-mission Gibbs chain file with Cls (for BR estimator)
            print(f"Copying {resamp} to {procver}/BP_resamp_c" + str(i).zfill(4) + f"_full_Cl_{procver}.h5")
            shutil.copyfile(resamp, f"{procver}/BP_resamp_c" + str(i).zfill(4) + f"_full_Cl_{procver}.h5",)

    """
    IQU mean, IQU stdev, (Masks for cmb)
    Run mean and stddev from min to max sample (Choose min manually or start at 1?)
    """
    chain = f"{procver}/BP_c0001_full_{procver}.h5"
    if freqmaps:
        try:
            # Full-mission 30 GHz IQU frequency map
            # BP_030_IQU_full_n0512_{procver}.fits
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_RMS", "Q_RMS", "U_RMS", "P_RMS",],
                units=["uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK",],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="030",
                fwhm=0.0,
                nu_ref_t="30.0 GHz",
                nu_ref_p="30.0 GHz",
                procver=procver,
                filename=f"BP_030_IQU_full_n0512_{procver}.fits",
                bndctr=30,
                restfreq=28.456,
                bndwid=9.899,
            )
            # Full-mission 44 GHz IQU frequency map
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_RMS", "Q_RMS", "U_RMS", "P_RMS",],
                units=["uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK",],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="044",
                fwhm=0.0,
                nu_ref_t="44.0 GHz",
                nu_ref_p="44.0 GHz",
                procver=procver,
                filename=f"BP_044_IQU_full_n0512_{procver}.fits",
                bndctr=44,
                restfreq=44.121,
                bndwid=10.719,
            )
            # Full-mission 70 GHz IQU frequency map
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_RMS", "Q_RMS", "U_RMS", "P_RMS",],
                units=["uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="070",
                fwhm=0.0,
                nu_ref_t="70.0 GHz",
                nu_ref_p="70.0 GHz",
                procver=procver,
                filename=f"BP_070_IQU_full_n1024_{procver}.fits",
                bndctr=70,
                restfreq=70.467,
                bndwid=14.909,
            )


        except Exception as e:
            print(e)
            print("Continuing...")

    
    if diff:
        import healpy as hp
        try:
            print("Creating frequency difference maps")
            path_dx12 = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/dx12"
            path_npipe = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/npipe"
            maps_dx12 = ["30ghz_2018_n1024_beamscaled_dip.fits","44ghz_2018_n1024_beamscaled_dip.fits","70ghz_2018_n1024_beamscaled_dip.fits"]
            maps_npipe = ["npipe6v20_030_map_uK.fits", "npipe6v20_044_map_uK.fits", "npipe6v20_070_map_uK.fits",]
            maps_BP = [f"BP_030_IQU_full_n0512_{procver}.fits", f"BP_044_IQU_full_n0512_{procver}.fits", f"BP_070_IQU_full_n1024_{procver}.fits",]
            beamscaling = [9.8961854E-01, 9.9757886E-01, 9.9113965E-01]
            for i, freq in enumerate(["030", "044", "070",]):
                map_BP    = hp.read_map(f"{procver}/{maps_BP[i]}", field=(0,1,2), verbose=False,)
                map_npipe = hp.read_map(f"{path_npipe}/{maps_npipe[i]}", field=(0,1,2), verbose=False,)
                map_dx12  = hp.read_map(f"{path_dx12}/{maps_dx12[i]}", field=(0,1,2), verbose=False,)
                
                #dx12 dipole values:
                # 3362.08 pm 0.99, 264.021 pm 0.011, 48.253 ± 0.005
                # 233.18308357  2226.43833645 -2508.42179665
                #dipole_dx12 = -3362.08*hp.dir2vec(264.021, 48.253, lonlat=True)

                #map_dx12  = map_dx12/beamscaling[i]
                # Smooth to 60 arcmin
                map_BP = hp.smoothing(map_BP, fwhm=arcmin2rad(60.0), verbose=False)
                map_npipe = hp.smoothing(map_npipe, fwhm=arcmin2rad(60.0), verbose=False)
                map_dx12 = hp.smoothing(map_dx12, fwhm=arcmin2rad(60.0), verbose=False)

                #ud_grade 30 and 44ghz
                if i<2:
                    map_npipe = hp.ud_grade(map_npipe, nside_out=512, verbose=False)
                    map_dx12 = hp.ud_grade(map_dx12, nside_out=512, verbose=False)

                # Remove monopoles
                map_BP -= np.mean(map_BP,axis=1).reshape(-1,1)
                map_npipe -= np.mean(map_npipe,axis=1).reshape(-1,1)
                map_dx12 -= np.mean(map_dx12,axis=1).reshape(-1,1)

                hp.write_map(f"{procver}/BP_{freq}_diff_npipe_{procver}.fits", np.array(map_BP-map_npipe), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"])
                hp.write_map(f"{procver}/BP_{freq}_diff_dx12_{procver}.fits", np.array(map_BP-map_dx12), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"])

        except Exception as e:
            print(e)
            print("Continuing...")

    if diffcmb:
        import healpy as hp
        try:
            print("Creating cmb difference maps")
            path_cmblegacy = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/cmb-legacy"
            mask_ = hp.read_map("/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits", verbose=False, dtype=np.bool,)
            map_BP = hp.read_map(f"{procver}/BP_cmb_IQU_full_n1024_{procver}.fits", field=(0,1,2), verbose=False, dtype=None,)
            map_BP_masked = hp.ma(map_BP[0])
            map_BP_masked.mask = np.logical_not(mask_)
            mono, dip = hp.fit_dipole(map_BP_masked)
            nside = 1024
            ray = range(hp.nside2npix(nside))
            vecs = hp.pix2vec(nside, ray)
            dipole = np.dot(dip, vecs)
            map_BP[0] = map_BP[0] - dipole - mono
            map_BP = hp.smoothing(map_BP, fwhm=arcmin2rad(np.sqrt(60.0**2-14**2)), verbose=False)
            #map_BP -= np.mean(map_BP,axis=1).reshape(-1,1)
            for i, method in enumerate(["commander", "sevem", "nilc", "smica",]):

                data = f"COM_CMB_IQU-{method}_2048_R3.00_full.fits"
                print(f"making difference map with {data}")
                map_cmblegacy  = hp.read_map(f"{path_cmblegacy}/{data}", field=(0,1,2), verbose=False,)
                map_cmblegacy = hp.smoothing(map_cmblegacy, fwhm=arcmin2rad(60.0), verbose=False)
                map_cmblegacy = hp.ud_grade(map_cmblegacy, nside_out=1024, verbose=False)
                map_cmblegacy = map_cmblegacy*1e6

                # Remove monopoles
                map_cmblegacy_masked = hp.ma(map_cmblegacy[0])
                map_cmblegacy_masked.mask = np.logical_not(mask_)
                mono = hp.fit_monopole(map_cmblegacy_masked)
                print(f"{method} subtracting monopole {mono}")
                map_cmblegacy[0] = map_cmblegacy[0] - mono #np.mean(map_cmblegacy,axis=1).reshape(-1,1)

                hp.write_map(f"{procver}/BP_cmb_diff_{method}_{procver}.fits", np.array(map_BP-map_cmblegacy), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"])

        except Exception as e:
            print(e)
            print("Continuing...")

    """
    FOREGROUND MAPS
    """
    # Full-mission CMB IQU map
    if cmb:
        fname = f"{procver}/BP_resamp_c0001_full_Cl_{procver}.h5" if resamp else chain
        try:
            format_fits(
                fname,
                extname="COMP-MAP-CMB",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS", "mask1", "mask2",],
                units=["uK_cmb", "uK_cmb", "uK", "uK", "NONE", "NONE",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="CMB",
                fwhm=14.0,
                nu_ref_t="NONE",
                nu_ref_p="NONE",
                procver=procver,
                filename=f"BP_cmb_IQU_full_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            print("Continuing...")

    if ff:
        try:
            # Full-mission free-free I map
            format_fits(
                chain,
                extname="COMP-MAP-FREE-FREE",
                types=["I_MEAN", "I_TE_MEAN", "I_RMS", "I_TE_RMS",],
                units=["uK_RJ", "K", "uK_RJ", "K",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=False,
                component="FREE-FREE",
                fwhm=30.0,
                nu_ref_t="40.0 GHz",
                nu_ref_p="40.0 GHz",
                procver=procver,
                filename=f"BP_freefree_I_full_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            print("Continuing...")

    if ame:
        try:
            # Full-mission AME I map
            format_fits(
                chain,
                extname="COMP-MAP-AME",
                types=["I_MEAN", "I_NU_P_MEAN", "I_RMS", "I_NU_P_RMS",],
                units=["uK_RJ", "GHz", "uK_RJ", "GHz",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=False,
                component="AME",
                fwhm=30.0,
                nu_ref_t="22.0 GHz",
                nu_ref_p="22.0 GHz",
                procver=procver,
                filename=f"BP_ame_I_full_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            print("Continuing...")

    if synch:
        try:
            # Full-mission synchrotron IQU map
            format_fits(
                chain,
                extname="COMP-MAP-SYNCHROTRON",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_BETA_MEAN", "QU_BETA_MEAN", "I_RMS", "Q_RMS", "U_RMS", "P_RMS", "I_BETA_RMS", "QU_BETA_RMS",],
                units=["uK_RJ", "uK_RJ", "uK_RJ", "uK_RJ", "NONE", "NONE", "uK_RJ","uK_RJ","uK_RJ","uK_RJ", "NONE", "NONE",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="SYNCHROTRON",
                fwhm=60.0,  # 60.0,
                nu_ref_t="30.0 GHz",
                nu_ref_p="30.0 GHz",
                procver=procver,
                filename=f"BP_synch_IQU_full_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            print("Continuing...")

    if dust:
        try:
            # Full-mission thermal dust IQU map
            format_fits(
                chain,
                extname="COMP-MAP-DUST",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_BETA_MEAN", "QU_BETA_MEAN", "I_T_MEAN", "QU_T_MEAN", "I_RMS", "Q_RMS", "U_RMS", "P_RMS", "I_BETA_RMS", "QU_BETA_RMS", "I_T_RMS", "QU_T_RMS",],
                units=["uK_RJ", "uK_RJ", "uK_RJ", "uK_RJ", "NONE", "NONE", "K", "K", "uK_RJ","uK_RJ","uK_RJ","uK_RJ", "NONE", "NONE", "K", "K",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="DUST",
                fwhm=10.0,  # 60.0,
                nu_ref_t="545 GHz",
                nu_ref_p="353 GHz",
                procver=procver,
                filename=f"BP_dust_IQU_full_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            print("Continuing...")

    """ As implemented by Simone
    """
    if br and resamp:
        # Gaussianized TT Blackwell-Rao input file
        print()
        print("{:-^50}".format("CMB GBR"))
        ctx.invoke(sigma_l2fits, filename=resamp, nchains=1, burnin=burnin, path="cmb/sigma_l", outname=f"{procver}/BP_cmb_GBRlike_{procver}.fits", save=True,)

    """
    TODO Generalize this so that they can be generated by Elina and Anna-Stiina
    """
    # Full-mission 30 GHz IQU beam symmetrized frequency map
    # BP_030_IQUdeconv_full_n0512_{procver}.fits
    # Full-mission 44 GHz IQU beam symmetrized frequency map
    # BP_044_IQUdeconv_full_n0512_{procver}.fits
    # Full-mission 70 GHz IQU beam symmetrized frequency map
    # BP_070_IQUdeconv_full_n1024_{procver}.fits

    """ Both sigma_l's and Dl's re in the h5. (Which one do we use?)
    """
    # CMB TT, TE, EE power spectrum
    # BP_cmb_Cl_{procver}.txt

    """ Just get this from somewhere
    """
    # Best-fit LCDM CMB TT, TE, EE power spectrum
    # BP_cmb_bfLCDM_{procver}.txt


@commands.command()
@click.argument('filename', type=click.STRING)
@click.option('-min', default=0, help='Min sample of dataset (burnin)')
@click.option('-max', default=1000, help='Max sample to inclue')
@click.option('-nbins', default=1, help='Bins')
def traceplot(filename, max, min, nbins):
    """
    This function plots a traceplot of samples from min to max with optional bins.
    Useful to plot sample progression of spectral indexes.
    """
    header = ['Prior', 'High lat.', 'NGS',
         'Gal. center', 'Fan region', 'Gal. anti-center',
         'Gum nebula']
    cols = [4,5,9,10,11,12,13]
    import pandas as pd
    df = pd.read_csv(filename, sep=r"\s+", usecols=cols, skiprows=range(min), nrows=max)
    df.columns = header
    x = 'MCMC Sample'
    
    traceplotter(df, header, x, nbins, outname=filename.replace(".dat","_traceplot.pdf"), min_=min)

@commands.command()
@click.argument("chainfile", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.option('-burnin', default=0, help='Min sample of dataset (burnin)')
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option('-plot', is_flag=True, default=False, help= 'Plots trace')
@click.option('-freeze', is_flag=True, help= 'Freeze top regions')
@click.option('-nbins', default=1, help='Bins for plotting')
def pixreg2trace(chainfile, dataset, burnin, maxchain, plot, freeze, nbins,):
    """
    Outputs the values of the pixel regions for each sample to a dat file.
    ex. pixreg2trace chain_c0001.h5 synch/beta_pixreg_val -burnin 30 -maxchain 4 
    """
    
    # Check if you want to output a map
    import h5py
    import healpy as hp
    import pandas as pd
    from tqdm import tqdm
    dats = []
    min_=burnin
    for c in range(1, maxchain + 1):
        chainfile_ = chainfile.replace("c0001", "c" + str(c).zfill(4))
        with h5py.File(chainfile_, "r") as f:
            max_ = len(f.keys()) - 1
            print("{:-^48}".format(f" Samples {min_} to {max_} in {chainfile_} "))
            for sample in tqdm(range(0, max_ + 1), ncols=80):
                # Identify dataset
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
                        print(f"WARNING! {tag} empty")
                    dats.append(data)
                except:
                    print(f"Found no dataset called {dataset}")
                    # Append sample to list
             

    sigs = ["T","P"]
    df = pd.DataFrame.from_records(dats, columns=sigs)
    header = ['Top left', 'Top right', 'Bot. left', 'Bot. right', 'NGS',
              'Gal. center', 'Fan region', 'Gal. anti-center',
              'Gum nebula']

    for sig in sigs:
        df2 = pd.DataFrame(df[sig].to_list(), columns=header)
        label = dataset.replace("/","-")
        outname = f"sampletrace_{sig}_{label}"
        df2.to_csv(f'{outname}.csv')
    
        if plot:
            xlabel = 'Gibbs Sample'
            if freeze:
                combined_hilat = 'High lat.'
                df2 = df2.drop(columns=['Top left', 'Top right', 'Bot. left',])
                df2 = df2.rename(columns={'Bot. right':combined_hilat})
                header_ = [combined_hilat] + header[4:]
            else:
                header_ = header.copy()

            traceplotter(df2, header_, xlabel, nbins, f"{outname}.pdf", min_=burnin*maxchain)

def traceplotter(df, header, xlabel, nbins, outname, min_):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.2})
    sns.set_style("whitegrid")
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    }
    sns.set_style(custom_style)

    #df.columns = y
    N = df.values.shape[0]
    df['Mean'] = df.mean(axis=1)
    df[xlabel] = range(N)
    header.append('Mean')

    f, ax = plt.subplots(figsize=(16,8))
    cmap = plt.cm.get_cmap('tab10')# len(y))

    means = df.mean()
    stds = df.std()
    # Reduce points
    if nbins>1:
        df = df.groupby(np.arange(len(df))//nbins).mean()

    positions = legend_positions(df, header)
    c = 0
    for i, (column, position) in enumerate(positions.items()):
        linestyle = '-'
        linewidth = 2
        fontweight = 'normal'
        if column == "Mean":
            color="grey"
            linewidth = 4
            fontweight='bold'
        else:
            color = cmap(c)#float(i-1)/len(positions))
            c += 1
        # Plot each line separatly so we can be explicit about color
        ax = df.plot(x=xlabel, y=column, legend=False, ax=ax, color=color, linestyle=linestyle, linewidth=linewidth)

        label = rf'{column} {means[i]:.2f}$\pm${stds[i]:.2f}'
        if len(label) > 24:
            label = f'{column} \n' + fr'{means[i]:.2f}$\pm${stds[i]:.2f}'

        # Add the text to the right
        plt.text(
            df[xlabel][df[column].last_valid_index()]+N*0.01,
            position, label, fontsize=12,
            color=color, fontweight=fontweight
        )

    #if min_:
    #    plt.xticks(list(plt.xticks()[0]) + [min_])

    ax.set_ylabel('Region spectral index')

    #ax.axes.xaxis.grid()
    #ax.axes.yaxis.grid()
    # Add percent signs
    #ax.set_yticklabels(['{:3.0f}%'.format(x) for x in ax.get_yticks()])
    sns.despine(top=True, right=True, left=True, bottom=True)

    #plt.xlim(min_, max_)
    plt.savefig(outname, dpi=300)
    plt.show()


@commands.command()
@click.argument("label", type=click.STRING)
@click.argument("freqs", type=click.FLOAT, nargs=-1)
@click.argument("nside", type=click.INT,)
@click.option("-cmb", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-synch", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-dust", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-ff", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-ame", type=click.Path(exists=True), help="Include resampled chain file",)
#@click.option("-skipcopy", is_flag=True, help="Don't copy full .h5 file",)
def generate_sky(label, freqs, nside, cmb, synch, dust, ff, ame):
    """
    Generate sky maps from separate input maps. 
    Reference frequencies from BP: CMB 1, SYNCH 30, DUST 545 353, FF 40, AME 22,
    Example:
    "c3pp generate-sky test 22 30 44 1024 -cmb BP_cmb_IQU_full_n1024_v1.0.fits -synch BP_synch_IQU_full_n1024_v1.0.fits"
    # Todo smoothing and nside?
    """
    import healpy as hp
    import numpy as np
    # Generate sky maps
    A = 1 # Only relative scaling
    for nu in freqs:
        filename = f"{label}_{nu}_n{nside}_generated.fits"
        print(f"Generating {filename}")
        data = np.zeros((3, hp.nside2npix(nside)))
        for pl in range(3):

            if cmb:
                pl_data  = hp.read_map(cmb, field=pl, verbose=False,)

            if synch:
                scaling  = fgs.lf(nu, A, betalf=-3.11, nuref=30.)
                print(hp.read_map(synch, field=pl, verbose=False)*scaling)
                pl_data += hp.read_map(synch, field=pl, verbose=False)*scaling

            if dust:
                if pl > 0:
                    scaling = fgs.dust(nu, A, beta=1.6, Td=18.5, nuref=353.,)
                else:
                    scaling = fgs.dust(nu, A, beta=1.6, Td=18.5, nuref=545.,)
                pl_data += hp.read_map(dust, field=pl, verbose=False,)*scaling

            if pl == 0:
                if ff:
                    scaling  = fgs.ff(nu, A, Te=7000., nuref=40.) 
                    pl_data += hp.read_map(ff, field=pl, verbose=False,)*scaling

                if ame:
                    scaling  = fgs.sdust(nu, A, nu_p=21, nuref=22.)
                    pl_data += hp.read_map(ame, field=pl, verbose=False,)*scaling

            data[pl] = pl_data
        hp.write_map(filename, data)
"""
def makespec():
    
    Implement spectrum as it appears on github.
    Spectral parameters can either be specified, in which case they will act fullsky, or maps can be given.
    
    for i in range(num_fg_comp):
        for j, nu in enumerate(frequencies):
            for sig in range(3):
                for p in range(npix):
                    map_[p, sig] = fg[i].amp[p,sig] * get_ideal_spectrum(fg[i].label, par_smooth(p, sig, i), nu)

                # Which mask to use
                for m in range(2):
                    mu     = np.sum(map_[:, sig]*mask[:, sig, m]) / np.sum(mask[:, sig, m])

                    f[nu,m] = np.log( np.sqrt(np.sum( (map_[:,sig]*mask[:,sig,m]-mu)**2) / (np.sum(mask[:,sig,m])-1.d0)) )
"""

@commands.command()
@click.argument('dir1', type=click.STRING)
@click.argument('type1', type=click.Choice(['ml', 'mean']))
@click.argument('dir2', type=click.STRING)
@click.argument('type2', type=click.Choice(['ml', 'mean']))
@click.pass_context
def make_diff_plots(ctx, dir1, dir2, type1, type2):
    '''
    Produces standard c3pp plots from the differences between
    two output directories given by dir1 and dir2
    '''

    comps = ['030', '044', '070', 'ame', 'cmb', 'freefree', 'synch']

    filenames = {dir1:'', dir2:''}

    import glob
    import healpy as hp


    for dirtype, dirloc in zip([type1, type2],[dir1, dir2]):
        if dirtype == 'ml':
            #determine largest sample number
            cmbs = glob.glob(os.path.join(dirloc, 'cmb_c0001_k??????.fits'))
            indexes = [int(cmb[-11:-5]) for cmb in cmbs]
            intdexes = [int(index) for index in indexes]
            index = max(intdexes)
            
            filenames[dirloc] = '_c0001_k' + str(index).zfill(6) + '.fits'

    for comp in comps:
        mapn = {dir1:'', dir2:''}

        for dirtype, dirloc in zip([type1, type2],[dir1, dir2]): 
            print(filenames[dirloc])
            if len(filenames[dirloc]) == 0:
                mapn[dirloc] = glob.glob(os.path.join(dirloc, 'BP_' + comp + '_I*.fits'))[0]
            else:
                if comp in ['ame', 'cmb', 'synch', 'dust']:
                    mapn[dirloc] = comp + filenames[dirloc]
                elif comp in ['freefree']:
                    mapn[dirloc] = 'ff' + filenames[dirloc]
                else:
                    mapn[dirloc] = 'tod_' + comp + '_map' + filenames[dirloc]
      
        print(mapn) 
        map1 = hp.read_map(os.path.join(dir1, mapn[dir1]))
        map2 = hp.read_map(os.path.join(dir2, mapn[dir2]))

        diff_map = map1 - map2 
  
        from c3postproc.plotter import Plotter
 
        Plotter(input=comp + '_diff' + '.fits', dataset='', nside=None, auto=True, min=None, max=None, mid=0.0,
                rng='auto', colorbar=True, lmax=None, fwhm=0.0, mask=None, mfill=None, sig=[0,], remove_dipole=None,
                logscale=None, size='m', white_background=True, darkmode=False, png=False, cmap=None, title=None,
                ltitle=None, unit=None, scale=1.0, outdir='.', verbose=False, data=diff_map)

@commands.command()
@click.option("-pol", is_flag=True, help="",)
@click.option("-long", is_flag=True, help="",)
@click.option("-darkmode", is_flag=True, help="",)
@click.option("-png", is_flag=True, help="",)
@click.option("-nside", type=click.INT, help="",)
@click.option("-a_cmb", help="",)
@click.option("-a_s",  help="",)
@click.option("-b_s",  help="",)
@click.option("-a_ff", help="",)
@click.option("-t_e",  help="",)
@click.option("-a_ame1",help="",)
@click.option("-a_ame2", help="",)
@click.option("-nup",  help="",)
@click.option("-a_d", help="",)
@click.option("-b_d", help="",)
@click.option("-t_d", help="",)
@click.option("-a_co10", help="",)
@click.option("-a_co21", help="",)
@click.option("-a_co32", help="",)
@click.option("-mask1",  help="",)
@click.option("-mask2",  help="",)
def output_sky_model(pol, long, darkmode, png, nside, a_cmb, a_s, b_s, a_ff, t_e, a_ame1, a_ame2, nup, a_d, b_d, t_d, a_co10, a_co21, a_co32, mask1, mask2):
    """
    Outputs spectrum plots
    c3pp output-sky-model -a_s synch_c0001_k000100.fits -b_s synch_beta_c0001_k000100.fits -a_d dust_init_kja_n1024.fits -b_d dust_beta_init_kja_n1024.fits -t_d dust_T_init_kja_n1024.fits -a_ame1 ame_c0001_k000100.fits -nup ame_nu_p_c0001_k000100.fits -a_ff ff_c0001_k000100.fits -t_e ff_Te_c0001_k000100.fits -mask1 mask_70GHz_t70.fits -mask2 mask_70GHz_t7.fits -nside 16
    """
    from c3postproc.spectrum import Spectrum
    """
    if not a_cmb:
        a_cmb = 0.67 if pol else 45
    if not a_s:
        a_s = 12 if pol else 76
    if not b_s:
        b_s = -3.1
    if not a_ff:
        a_ff = 30.
    if not t_e:
        t_e = 7000.
    if not a_ame1:
        a_ame1 = 5 if pol else 50
    if not a_ame2:
        a_ame2 = 50.
    if not nup:
        nup = 24
    if not a_d:
        a_d = 8 if pol else 163
    if not b_d:
        b_d = 1.6
    if not t_d:
        t_d = 18.5
    if not a_co10:
        a_co10=50
    if not a_co21:
        a_co21=25
    if not a_co32:
        a_co32=10
    """

    if pol:
        p = 1.5 if long else 12
        foregrounds = {
            "CMB EE":       {"function": "rspectrum", 
                             "params"  : [1, "EE"],
                             "position": p,
                             "color"   : "C5",
                             "sum"     : False,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Synchrotron" : {"function": "lf", 
                             "params"  : [a_s, b_s,],
                             "position": 15,
                             "color"   : "C2",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Thermal Dust": {"function": "tdust", 
                             "params": [a_d, b_d, t_d, 353],
                             "position": 150,
                             "color":    "C1",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         }, 
            "Sum fg."      : {"function": "sum", 
                             "params"  : [],
                             "position": 45,
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "--",
                             "gradient": False,
                          },
            r"BB $r=0.001$"   :  {"function": "rspectrum", 
                             "params"  : [0.01, "BB",],
                             "position": p,
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "dotted",
                             "gradient": True,
                         },
            r"BB $r=0.0001$"   :  {"function": "rspectrum", 
                             "params"  : [1e-4, "BB",],
                             "position": p,
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "dotted",
                             "gradient": True,
                         },

            }
    else:
        p = 3 if long else 57
        foregrounds = {
            "CMB":          {"function": "rspectrum", 
                             "params"  : [1., "TT"],
                             "position": 70,
                             "color"   : "C5",
                             "sum"     : False,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Synchrotron" : {"function": "lf", 
                             "params"  : [a_s, b_s,],
                             "position": 120,
                             "color"   : "C2",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Thermal Dust": {"function": "tdust", 
                             "params": [a_d, b_d, t_d, 545],
                             "position": 12,
                             "color":    "C1",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         }, 
            "Free-Free"  : {"function": "ff", 
                             "params"  : [a_ff, t_e,],
                             "position": 40,
                             "color"   : "C0",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Spinning dust" : {"function": "sdust", 
                             "params"  : [a_ame1, nup,],
                             "position": p,
                             "color"   : "C4",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            r"CO$_{1\rightarrow 0}$": {"function": "line", 
                                       "params"  : [a_co10, 115, 11.06],
                                       "position": p,
                                       "color"   : "C9",
                                       "sum"     : True,
                                       "linestyle": "solid",
                                       "gradient": False,
                         },
            r"CO$_{2\rightarrow 1}$": {"function": "line", 
                                       "params"  : [a_co21, 230., 14.01],
                                       "position": p,
                                       "color"   : "C9",
                                       "sum"     : True,
                                       "linestyle": "solid",
                                       "gradient": False,
                         },
            r"CO$_{3\rightarrow 2}$":      {"function": "line", 
                                            "params"  : [a_co32, 345., 12.24],
                                            "position": p,
                                            "color"   : "C9",
                                            "sum"     : True,
                                            "linestyle": "solid",
                                            "gradient": False,
                         },
            "Sum fg."      : {"function": "sum", 
                             "params"  : [],
                             "position": 20,
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "--",
                             "gradient": False,
                          },
            }


    Spectrum(pol, long, darkmode, png, foregrounds, [mask1,mask2], nside)



