import time
import os
import numpy as np
import sys
import click
from src.tools import *


@click.group()
def commands_plotting():
    pass

@commands_plotting.command()
@click.argument("input", type=click.STRING)
def specplot(input,):
    """
    This function plots the file output by the Crosspec function.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from cycler import cycler
    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
    sns.set_style("whitegrid")
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi':300,
        'font.size': 20, 
        'axes.linewidth': 1.5,
        'axes.prop_cycle': cycler(color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])
    }
    sns.set_style(custom_style)
    lmax = 200
    ell, ee, bb, eb = np.loadtxt(input, usecols=(0,2,3,6), skiprows=3, max_rows=lmax, unpack=True)
                                 
    ee = ee*(ell*(ell+1)/(2*np.pi))
    bb = bb*(ell*(ell+1)/(2*np.pi))
    eb = eb*(ell*(ell+1)/(2*np.pi))
	
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    l1 = ax1.loglog(ell, ee, linewidth=2, label="EE", color='#636EFA')
    l2 = ax1.loglog(ell, bb, linewidth=2, label="BB", color= '#EF553B')
    ax1.set_ylabel(r"$D_l$ [$\mu K^2$]",)

    l3 = ax2.semilogx(ell, bb/ee, linewidth=2, label="BB/EE", color='#00CC96')
    ax2.set_ylabel(r"BB/EE",)
    ax2.set_xlabel(r'Multipole moment, $l$',)
    #plt.semilogx(ell, eb, label="EB")
	
    sns.despine(top=True, right=True, left=True, bottom=False, ax=ax1)
    sns.despine(top=True, right=True, left=True, bottom=True, ax=ax2)
    #plt.xlim(0,200)
    ax1.set_ylim(0.11,150)
    ax2.set_ylim(-0.,2.)
    #ax.axes.xaxis.grid()
    ls = l1+l2+l3
    labs = [l.get_label() for l in ls]
    ax1.legend(ls, labs, frameon=False,)
    #ax1.legend(frameon=False)
    plt.tight_layout(h_pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.savefig(input.replace(".dat",".pdf"), dpi=300)
    plt.show()

@commands_plotting.command()
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
@click.option("-fwhm", default=0.0, type=click.FLOAT, help="FWHM of smoothing, in arcmin.",)
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
    RECOMMENDED: Use -auto to autodetect map type and set parameters.\n
    Some autodetected maps use logscale, you will be warned.
    """
    if input.endswith(".h5") and not dataset and not nside:
        print("Specify Nside when plotting alms!")
        sys.exit()
        
    data = None # Only used if calling plotter directly for plotting data array
    from src.plotter import Plotter
    Plotter(input, dataset, nside, auto, min, max, mid, range, colorbar, lmax, fwhm, mask, mfill, sig, remove_dipole, logscale, size, white_background, darkmode, png, cmap, title, ltitle, unit, scale, outdir, verbose,data)


@commands_plotting.command()
@click.argument("filename", type=click.STRING)
@click.argument("lon", type=click.INT)
@click.argument("lat", type=click.INT)
@click.argument("size", type=click.INT)
@click.option("-sig", default=0, help="Which sky signal to plot",)
@click.option("-min", "min_", help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", "max_", help="Max value of colorbar, overrides autodetector.",)
@click.option("-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-cmap", default="planck", help="Choose different color map (string), such as Jet or planck",)
@click.option("-graticule", is_flag=True, help="Add graticule",)
@click.option("-log", is_flag=True, help="Add graticule",)
@click.option("-nobar", is_flag=True, help="remove colorbar",)
@click.option("-outname", help="Output filename, else, filename with different format.",)
def gnomplot(filename, lon, lat, sig, size, min_, max_, unit, cmap, graticule, log, nobar, outname):
    import healpy as hp
    import matplotlib.pyplot as plt
    from src.plotter import fmt
    from functools import partial
    
    from matplotlib import rcParams, rc
    rcParams["backend"] = "pdf"
    rcParams["legend.fancybox"] = True
    rcParams["lines.linewidth"] = 2
    rcParams["savefig.dpi"] = 300
    rcParams["axes.linewidth"] = 1
    rc("text.latex", preamble=r"\usepackage{sfmath}",)

    if cmap == "planck":
        import matplotlib.colors as col
        from pathlib import Path
        cmap = Path(__file__).parent / "parchment1.dat"
        cmap = col.ListedColormap(np.loadtxt(cmap) / 255.0, "planck")
    else:
        try:
            import cmasher
            cmap = eval(f"cmasher.{cmap}")
        except:
            cmap = plt.get_cmap(cmap)

    xsize = 5000
    reso = size*60/xsize
    fontsize=10
    x = hp.read_map(filename, field=sig, verbose=False, dtype=None)
    nside=hp.get_nside(x)

    half = size/2
    proj = hp.projector.CartesianProj(lonra=[lon-half,lon+half], latra=[lat-half, lat+half], coord='G', xsize=xsize, ysize=xsize)
    reproj_im = proj.projmap(x, vec2pix_func=partial(hp.vec2pix, nside))

    #norm="log" if log else None
    image = plt.imshow(reproj_im, origin='lower', interpolation='nearest', vmin=min_,vmax=max_, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    if not nobar:
        # colorbar
        from matplotlib.ticker import FuncFormatter
        cb = plt.colorbar(image, orientation="horizontal", shrink=0.5, pad=0.03, format=FuncFormatter(fmt))
        cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize)
        cb.ax.xaxis.set_label_text(unit)
        cb.ax.xaxis.label.set_size(fontsize+2)

    if graticule:
        hp.graticule()
    if not outname:
        outname = filename.replace(".fits", ".pdf")

    plt.savefig(outname, bbox_inches="tight", pad_inches=0.02, transparent=True, format="pdf",)


@commands_plotting.command()
@click.argument("procver", type=click.STRING)
@click.option("-mask", type=click.Path(exists=True), help="Mask for calculating cmb",)
@click.option("-defaultmask", is_flag=True, help="Use default dx12 mask",)
@click.option("-freqmaps", is_flag=True, help=" output freqmaps",)
@click.option("-cmb", is_flag=True, help=" output cmb",)
@click.option("-cmbresamp", is_flag=True, help=" output cmbresamp",)
@click.option("-synch", is_flag=True, help=" output synch",)
@click.option("-ame", is_flag=True, help=" output ame",)
@click.option("-ff", is_flag=True, help=" output ff",)
@click.option("-dust", is_flag=True, help=" output dust",)
@click.option("-diff", is_flag=True, help="Creates diff maps to dx12 and npipe")
@click.option("-diffcmb", is_flag=True, help="Creates diff maps with cmb maps")
@click.option("-spec", is_flag=True, help="Creates emission plot")
@click.option("-all", "all_", is_flag=True, help="Output all")
@click.pass_context
def plotrelease(ctx, procver, mask, defaultmask, freqmaps, cmb, cmbresamp, synch, ame, ff, dust, diff, diffcmb, spec, all_):
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
            if (cmbresamp and mask) or (cmbresamp and defaultmask):
                outdir = "figs/cmb/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                if defaultmask:
                    mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"

                try:
                    # CMB I with dip
                    ctx.invoke(plot, input=f"BP_cmb_resamp_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, range=3400)
                    # CMB I without dip
                    ctx.invoke(plot, input=f"BP_cmb_resamp_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask,)
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if (cmb and mask) or (cmb and defaultmask):
                outdir = "figs/cmb/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                if defaultmask:
                    mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"
                    
                try:
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
                    ctx.invoke(plot, input=f"BP_cmb_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3, 4, 5,], )
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if freqmaps:
                outdir = "figs/freqmaps/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                try:
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
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if synch:
                outdir = "figs/synchrotron/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                try:
                    # Synch IQU
                    ctx.invoke(plot, input=f"BP_synch_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], )
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if ff:
                outdir = "figs/freefree/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                try:
                    # freefree mean and rms
                    ctx.invoke(plot, input=f"BP_freefree_I_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3], )
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if ame:
                outdir = "figs/ame/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                try:
                    # ame mean and rms
                    ctx.invoke(plot, input=f"BP_ame_I_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3], )
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if dust:
                outdir = "figs/dust/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
    
                try:
                    # dust IQU
                    ctx.invoke(plot, input=f"BP_dust_IQU_full_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], )
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if diff:
                outdir = "figs/freqmap_difference/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                    
                try:
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
                except Exception as e:
                    print(e)
                    print("Continuing...")

            if diffcmb:
                outdir = "figs/cmb_difference/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"
                for i, method in enumerate(["Commander", "SEVEM", "NILC", "SMICA",]):
                    try:
                        input = f"BP_cmb_diff_{method.lower()}_{procver}.fits"
                        ctx.invoke(plot, input=input, size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask, sig=[0,],  range=10, title=method, ltitle=" ",)
                        ctx.invoke(plot, input=input, size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4, title=method, ltitle=" ",)
                    except Exception as e:
                        print(e)
                        print("Continuing...")
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

        for long in [False,True]:
            for pol in [True,False]:
                ctx.invoke(output_sky_model, pol=pol, long=long,
                           darkmode=False, png=False,
                           nside=64, a_cmb=a_cmb, a_s=a_s, b_s=b_s, a_ff=a_ff,
                           t_e=t_e, a_ame1=a_ame1, a_ame2=a_ame2, nup=nup, a_d=a_d, b_d=b_d,
                           t_d=t_d, a_co10=a_co10, a_co21=a_co21, a_co32=a_co32, mask1=mask1,
                           mask2=mask2,)
        
        outdir = "figs/sky-model/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        files = os.listdir(".")
        for f in files:
            if f.startswith("spectrum"):
                os.rename(f, f"{outdir}{f}")

@commands_plotting.command()
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

@commands_plotting.command()
@click.argument("chainfile", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.option('-burnin', default=0, help='Min sample of dataset (burnin)')
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option('-plot', is_flag=True, default=False, help= 'Plots trace')
@click.option('-freeze', is_flag=True, help= 'Freeze top regions')
@click.option('-nbins', default=1, help='Bins for plotting')
@click.option("-f", "priorsamp",  multiple=True, help="These are sampled around prior and will be marked",)
@click.option('-scale', default=0.023, help='scale factor for labels')
def pixreg2trace(chainfile, dataset, burnin, maxchain, plot, freeze, nbins, priorsamp, scale):
    """
    Outputs the values of the pixel regions for each sample to a dat file.
    ex. c3pp pixreg2trace chain_c0001.h5 synch/beta_pixreg_val -burnin 30 -maxchain 4 
    """
    
    # Check if you want to output a map
    import h5py
    import healpy as hp
    import pandas as pd
    from tqdm import tqdm
    dats = []
    for c in range(1, maxchain + 1):
        chainfile_ = chainfile.replace("c0001", "c" + str(c).zfill(4))
        min_=burnin if c>1 else 0
        with h5py.File(chainfile_, "r") as f:
            max_ = len(f.keys()) - 1
            print("{:-^48}".format(f" Samples {min_} to {max_} in {chainfile_} "))
            for sample in tqdm(range(min_, max_ + 1), ncols=80):
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
    nregs = len(df["P"][0])
    if nregs == 9:
        header = ['Top left', 'Top right', 'Bot. left', 'Bot. right', 'NGS',
                  'Gal. center', 'Fan region', 'Gal. anti-center',
                  'Gum nebula']
    elif nregs == 6:
        header = ['High lat.', 'NGS',
                  'Gal. center', 'Fan region', 'Gal. anti-center',
                  'Gum nebula']
    elif nregs == 4:
        header = ['High lat.', 'NGS',
                  'Gal. center', 'Gal. plane']
    else:
        print("Number of columns not supported", nregs)
        sys.exit()
    

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

            traceplotter(df2, header_, xlabel, nbins, f"{outname}.pdf", min_=burnin, priorsamp=priorsamp, scale=scale)

def traceplotter(df, header, xlabel, nbins, outname, min_, priorsamp=None, scale=0.023):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.2})
    sns.set_style("whitegrid")
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi':300,
        'font.size': 20, 
        'axes.linewidth': 1.5,
    }
    sns.set_style(custom_style)

    #df.columns = y
    N = df.values.shape[0]
    nregs = len(header)
    """
    df['Mean'] = df.mean(axis=1)
    header.append('Mean')
    """
    df[xlabel] = range(N)
    f, ax = plt.subplots(figsize=(16,8))
    
    #cmap = plt.cm.get_cmap('tab20')# len(y))
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(['#636EFA', '#EF553B', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FECB52', '#FF97FF',  '#AB63FA',])
    #cmap = plt.cm.get_cmap('tab10')# len(y))

    means = df[min_:].mean()
    stds = df[min_:].std()
    # Reduce points
    if nbins>1:
        df = df.groupby(np.arange(len(df))//nbins).mean()
    
    #longestlab = max([len(x) for x in header])
    # scale = 0.075 for 9 regions worked well.
    positions = legend_positions(df, header, scaling=scale)
    c = 0
    for i, (column, position) in enumerate(positions.items()):
        linestyle = ':' if str(i) in priorsamp else '-'
        linewidth = 2
        fontweight = 'normal'
        if column == "Mean":
            color="#a6a6a6" #"grey"
            linewidth = 4
            fontweight='bold'
        else:
            color = cmap(c)#float(i-1)/len(positions))
            c += 1
        # Plot each line separatly so we can be explicit about color
        ax = df.plot(x=xlabel, y=column, legend=False, ax=ax, color=color, linestyle=linestyle, linewidth=linewidth,)
        """
        label = rf'{column} {means[i]:.2f}$\pm${stds[i]:.2f}'
        #if len(label) > 1: #24:
        #    label = f'{column} \n' + fr'{means[i]:.2f}$\pm${stds[i]:.2f}'

        # Add the text to the right
        plt.text(
            df[xlabel][df[column].last_valid_index()]+N*0.01,
            position, label, fontsize=15,
            color=color, fontweight=fontweight
        )
        """
        label1 = rf'{column}'
        label2 = rf'{means[i]:.2f}$\pm${stds[i]:.2f}'
        #if len(label) > 1: #24:
        #    label = f'{column} \n' + fr'{means[i]:.2f}$\pm${stds[i]:.2f}'

        # Add the text to the right
        plt.text(
            df[xlabel][df[column].last_valid_index()]+N*0.01,
            position, label1, fontsize=15,
            color=color, fontweight=fontweight
        )
        if nregs > 4:
            r = 0.16
        else:
            r = 0.12
        plt.text(
            df[xlabel][df[column].last_valid_index()]+N*r,
            position, label2, fontsize=15,
            color=color, fontweight='normal'
        )

    #if min_:
    #    plt.xticks(list(plt.xticks()[0]) + [min_])

    ax.set_ylabel('Region spectral index')
    #plt.yticks(rotation=90)
    plt.gca().set_xlim(right=N)
    #ax.axes.xaxis.grid()
    #ax.axes.yaxis.grid()
    # Add percent signs
    #ax.set_yticklabels(['{:3.0f}%'.format(x) for x in ax.get_yticks()])
    sns.despine(top=True, right=True, left=True, bottom=True)
    plt.subplots_adjust(wspace=0, hspace=0.01, right=0.81)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.show()


@commands_plotting.command()
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
  
        from src.plotter import Plotter
 
        Plotter(input=comp + '_diff' + '.fits', dataset='', nside=None, auto=True, min=None, max=None, mid=0.0,
                rng='auto', colorbar=True, lmax=None, fwhm=0.0, mask=None, mfill=None, sig=[0,], remove_dipole=None,
                logscale=None, size='m', white_background=True, darkmode=False, png=False, cmap=None, title=None,
                ltitle=None, unit=None, scale=1.0, outdir='.', verbose=False, data=diff_map)

@commands_plotting.command()
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
    from src.spectrum import Spectrum
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
            r"BB $r=0.01$"   :  {"function": "rspectrum", 
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



