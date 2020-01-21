import time
totaltime = time.time()
import sys
import os
import re
import healpy as hp
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.ticker as ticker
from matplotlib import rcParams, rc
print("Importtime:", (time.time() - totaltime))

def Plotter(
    input,
    dataset,
    nside,
    auto,
    min,
    max,
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


    rcParams["backend"] = "pdf"
    rcParams["legend.fancybox"] = True
    rcParams["lines.linewidth"] = 2
    rcParams["savefig.dpi"] = 300
    rcParams["axes.linewidth"] = 1
    # use of Sans Serif also in math mode
    masked = False

    if darkmode:
        rcParams["text.color"] = "white"  # axes background color
        rcParams["axes.facecolor"] = "white"  # axes background color
        rcParams["axes.edgecolor"] = "white"  # axes edge color
        rcParams["axes.labelcolor"] = "white"
        rcParams["xtick.color"] = "white"  # color of the tick labels
        rcParams["ytick.color"] = "white"  # color of the tick labels
        rcParams["grid.color"] = "white"  # grid color
        rcParams["legend.facecolor"] = "inherit"  # legend background color (when 'inherit' uses axes.facecolor)
        rcParams["legend.edgecolor"] = "white"  # legend edge color (when 'inherit' uses axes.edgecolor)

    rc("text.latex", preamble=r"\usepackage{sfmath}")

    # Which signal to plot
    pollist = get_pollist(sig)
    signal_labels = ["I", "Q", "U"]
    print("----------------------------------")
    print("Plotting " + input)

    #######################
    ####   READ MAP   #####
    #######################
    starttime = time.time()

    if input.endswith(".fits"):
        # Make sure its the right shape for indexing
        # This is a really dumb way of doing it
        dats = [0, 0, 0]
        map_exists = False
        for i in pollist:
            try:
                dats[i] = hp.ma(hp.read_map(input, field=i, verbose=False))
                nside = hp.npix2nside(len(dats[i]))
                map_exists = True
            except:
                print("Signal {} not found in data, skipping".format(signal_labels[i]))
                continue

        if map_exists == False:
            hihi = []
            [hihi.append(signal_labels[i]) for i in pollist]
            print()
            print("{} does not contain a {} signal. Breaking.".format(input, hihi))
            sys.exit()

        maps = np.array(dats)
        outfile = input.replace(".fits", "")
        
    elif input.endswith(".h5"):
        from c3postproc.commands import alm2fits_tool, h5map2fits

        if dataset.endswith("alm"):
            print("Converting alms to map")
            maps, nside, lmax, fwhm, outfile = alm2fits_tool(input, dataset, nside, lmax, fwhm, save=False)

        elif dataset.endswith("map"):
            print("Reading map from h5")
            maps, nside, lmax, outfile = h5map2fits(input, dataset, save=False)

        else:
            print("Dataset not found. Breaking.")
            print("Does {}/{} exist?".format(input, dataset))
            sys.exit()
    else:
        print("Dataset not found. Breaking.")
        sys.exit()

    print("Map reading: ", (time.time() - starttime)) if verbose else None
    print("nside", nside, "total file shape", maps.shape)

    for polt in pollist:
        m = maps[polt]  # Select map signal (I,Q,U)

        #######################
        #### Auto-param   #####
        #######################
        # ttl, unt and cmb are temporary variables for title, unit and colormap
        if auto:
            ttl, ticks, ticklabels, unt, cmp, lgscale = get_params(m, outfile, polt, signal_labels)
        else:
            ttl = ""
            unt = ""
            rng = "auto"
            ticks = [False, False]
            ticklabels = [False, False]
            cmp = "planck"
            lgscale = False



        # If range has been specified, set.
        if rng:
            if rng == "auto":
                mx = np.percentile(m, 95)
                mn = np.percentile(m, 5)
                print("Autocalculating limits, min {}, max {}".format(mn,mx))
                print("Manual limints, min {}, max {}".format(min, max))
                if min is False:
                    min = mn
                if max is False:
                    max = mx
                print("Limits after test, min {}, max {}".format(min, max))
            else:
                rng = float(rng)
                min = -rng
                max = rng

        # If min and max have been specified, set.
        if min is not False:
            ticks[0] = float(min)
            ticklabels[0] = str(min)

        if max is not False:
            ticks[-1] = float(max)
            ticklabels[-1] = str(max)

        print("max, max {}".format(ticks))
        print("tmax, tmin {}".format(ticklabels)) 
        
        ##########################
        #### Plotting Params #####
        ##########################
       
        # Upper right title
        if not title:
            title = ttl

        # Unit under colorbar
        if not unit:
            unit = unt
        
        # Image size -  ratio is always 1/2
        xsize = 2000
        ysize = int(xsize / 2.0)

        ########################
        #### remove dipole #####
        ########################
        if remove_dipole:
            starttime = time.time()
            print("Removing dipole")
            dip_mask_name = remove_dipole
            # Mask map for dipole estimation
            m_masked = hp.ma(m)
            m_masked.mask = np.logical_not(hp.read_map(dip_mask_name))
            
            # Fit dipole to masked map
            mono, dip = hp.fit_dipole(m_masked)
            print("Dipole vector: {}".format(dip))
            print("Dipole amplitude: {}".format(np.sqrt(np.sum(dip ** 2))))

            # Create dipole template
            nside = int(nside)
            ray = range(hp.nside2npix(nside))
            vecs = hp.pix2vec(nside, ray)
            dipole = np.dot(dip, vecs)

            # Subtract dipole map from data
            m = m - dipole
            print("Dipole removal : ", (time.time() - starttime)) if verbose else None
        #######################
        ####   logscale   #####
        #######################
        # Some maps turns on logscale automatically
        # -logscale command overwrites this
        if logscale == None:
            logscale = lgscale

        if logscale:
            starttime = time.time()

            m = np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))
            m = np.maximum(np.minimum(m, ticks[-1]), ticks[0])

            print("Logscale", (time.time() - starttime)) if verbose else None
        ######################
        #### COLOR SETUP #####
        ######################
        # Chose colormap manually
        if cmap:
            print("Setting colormap to {}".format(cmap))
            if cmap == "planck":
                from pathlib import Path

                cmap = Path(__file__).parent / "parchment1.dat"
                cmap = col.ListedColormap(np.loadtxt(cmap) / 255.0)
            else:
                cmap = plt.get_cmap(cmap)
        else:
            cmap = cmp

        #######################
        ####  Projection? #####
        #######################
        theta = np.linspace(np.pi, 0, ysize)
        phi = np.linspace(-np.pi, np.pi, xsize)
        longitude = np.radians(np.linspace(-180, 180, xsize))
        latitude = np.radians(np.linspace(-90, 90, ysize))

        # project the map to a rectangular matrix xsize x ysize
        PHI, THETA = np.meshgrid(phi, theta)
        grid_pix = hp.ang2pix(nside, THETA, PHI)

        ######################
        ######## Mask ########
        ######################
        if mask:
            print("Masking using {}".format(mask))
            masked = True

            # Apply mask
            hp.ma(m)
            m.mask = np.logical_not(hp.read_map(mask))
 
            # Don't know what this does, from paperplots by Zonca.
            grid_mask = m.mask[grid_pix]
            grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)
            
            if mfill:
                cmap.set_bad(mfill)  # color of missing pixels
                # cmap.set_under("white") # color of background, necessary if you want to use
                # using directly matplotlib instead of mollview has higher quality output
        else:
            grid_map = m[grid_pix]

        ######################
        #### Formatting ######
        ######################
        from matplotlib.projections.geo import GeoAxes

        class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
            """Shifts labelling by pi
            Shifts labelling from -180,180 to 0-360"""

            def __call__(self, x, pos=None):
                if x != 0:
                    x *= -1
                if x < 0:
                    x += 2 * np.pi
                return GeoAxes.ThetaFormatter.__call__(self, x, pos)

        sizes = get_sizes(size)
        for width in sizes:
            print("Plotting size " + str(width))
            height = width / 2.0

            # Make sure text doesnt change with colorbar
            height *= 1.275 if colorbar else 1

            ################
            ##### font #####
            ################
            if width > 12.0:
                fontsize = 8
            elif width == 12.0:
                fontsize = 7
            else:
                fontsize = 6

            fig = plt.figure(figsize=(cm2inch(width), cm2inch(height)))

            ax = fig.add_subplot(111, projection="mollweide")

            # rasterized makes the map bitmap while the labels remain vectorial
            # flip longitude to the astro convention
            image = plt.pcolormesh(
                longitude[::-1], latitude, grid_map, vmin=ticks[0], vmax=ticks[-1], rasterized=True, cmap=cmap
            )
            # graticule
            ax.set_longitude_grid(60)
            ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(60))

            if width < 10:
                ax.set_latitude_grid(45)
                ax.set_longitude_grid_ends(90)

            ################
            ### COLORBAR ###
            ################
            if colorbar:
                # colorbar
                cb = fig.colorbar(
                    image, orientation="horizontal", shrink=0.3, pad=0.08, ticks=ticks, format=ticker.FuncFormatter(fmt)
                )
                # Format tick labels if autosetting
                #if auto:
                #    cb.ax.set_xticklabels(ticklabels)
                cb.ax.xaxis.set_label_text(unit)
                cb.ax.xaxis.label.set_size(fontsize)
                cb.ax.minorticks_on()
                cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize)
                cb.ax.xaxis.labelpad = 4  # -11
                # workaround for issue with viewers, see colorbar docstring
                cb.solids.set_edgecolor("face")

            # remove longitude tick labels
            ax.xaxis.set_ticklabels([])
            # remove horizontal grid
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
            plt.grid(True)

            #############
            ## TITLE ####
            #############
            plt.text(6.0, 1.3, r"%s" % title, ha="center", va="center", fontsize=fontsize)

            ##############
            #### SAVE ####
            ##############
            plt.tight_layout()
            filetype = ".pdf" if pdf else ".png"
            tp = False if white_background else True  # Turn on transparency unless told otherwise

            ##############
            ## filename ##
            ##############
            starttime = time.time()
            filename = []
            filename.append("{}arcmin".format(str(int(fwhm)))) if fwhm > 0 else None
            filename.append("cb") if colorbar else None
            filename.append("masked") if masked else None
            filename.append("dark") if darkmode else None

            nside_tag = "_n" + str(int(nside))
            if nside_tag in outfile:
                outfile = outfile.replace(nside_tag, "")
            fn = outfile + "_" + signal_labels[polt] + "_w" + str(int(width)) + nside_tag

            for i in filename:
                fn += "_" + i
            fn += filetype

            plt.savefig(fn, bbox_inches="tight", pad_inches=0.02, transparent=tp)
            plt.close()
            print("Outputting", (time.time() - starttime)) if verbose else None
            print("Totaltime:", (time.time() - totaltime)) if verbose else None

def get_params(m, outfile, polt, signal_labels):
    print()
    logscale = False

    # Everything listed here will be recognized
    # If tag is found in output file, use template
    cmb_tags = ["cmb"]
    chisq_tags = ["chisq"]
    synch_tags = ["synch_c", "synch_amp"]
    dust_tags = ["dust_c", "dust_amp"]
    ame_tags = ["ame_c", "ame_amp", "ame1_c", "ame1_amp"]
    ff_tags = ["ff_c", "ff_amp"]
    co10_tags = ["co10", "co-100"]
    co21_tags = ["co21", "co-217"]
    co32_tags = ["co32", "co-353"]
    hcn_tags = ["hcn"]
    dust_T_tags = ["dust_T", "dust_Td"]
    dust_beta_tags = ["dust_beta"]
    synch_beta_tags = ["synch_beta"]
    ff_Te_tags = ["ff_T_e", "ff_Te"]
    ff_EM_tags = ["ff_EM"]
    res_tags = ["residual_", "res_"]
    tod_tags = ["Smap"]
    ignore_tags = ["radio_"]

    if tag_lookup(cmb_tags, outfile):
        print("----------------------------------")
        print("Plotting CMB " + signal_labels[polt])

        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{CMB}}$"

        if polt > 0:
            vmin = -2
            vmid = 0
            vmax = 2
        else:
            vmin = -300
            vmid = 0
            vmax = 300

        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{CMB}}$"

        from pathlib import Path

        color = Path(__file__).parent / "parchment1.dat"
        cmap = col.ListedColormap(np.loadtxt(color) / 255.0)

    elif tag_lookup(chisq_tags, outfile):
        title = r"$\chi^2$ " + signal_labels[polt]

        if polt > 0:
            vmin = 0
            vmax = 32
        else:
            vmin = 0
            vmax = 76

        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        print("----------------------------------")
        print("Plotting chisq with vmax = " + str(vmax) + " " + signal_labels[polt])

        unit = ""
        cmap = col.LinearSegmentedColormap.from_list("own2", ["black", "white"])


    elif tag_lookup(synch_tags, outfile):
        print("----------------------------------")
        print("Plotting Synchrotron" + " " + signal_labels[polt])
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{s}}$ "
        print("Applying logscale (Rewrite if not)")
        if polt > 0:
            vmin = -np.log10(50)
            vmax = np.log10(50)
            tmin = str(-50)
            tmax = str(50)
            logscale = True

            vmid = 0
            tmid = "0"
            ticks = [vmin, vmid, vmax]
            ticklabels = [tmin, tmid, tmax]

            col1 = "darkgoldenrod"
            col2 = "darkgreen"
            cmap = col.LinearSegmentedColormap.from_list("own2", [endcolor, col1, startcolor, col2, endcolor])
        else:
            vmin = 1
            vmax = np.log10(100)
            tmin = str(10)
            tmax = str(100)
            logscale = True
            cmap = col.LinearSegmentedColormap.from_list("own2", ["black", "green", "white"])

            ticks = [vmin, vmax]
            ticklabels = [tmin, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"

    elif tag_lookup(ff_tags, outfile):
        print("----------------------------------")
        print("Plotting freefree")
        print("Applying logscale (Rewrite if not)")

        vmin = 0  # 0
        vmid = np.log10(100)
        vmax = np.log10(10000)  # 1000

        tmin = str(0)
        tmid = str(r"$10^2$")
        tmax = str(r"$10^4$")

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{ff}}$"
        logscale = True
        cmap = col.LinearSegmentedColormap.from_list("own2", ["black", "Navy", "white"])


    elif tag_lookup(dust_tags, outfile):
        print("----------------------------------")
        print("Plotting Thermal dust" + " " + signal_labels[polt])
        print("Applying logscale (Rewrite if not)")
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{d}}$ "
        if polt > 0:
            vmin = -np.log10(100)
            vmid = 0
            vmax = np.log10(100)

            tmin = str(-100)
            tmid = 0
            tmax = str(100)

            logscale = True

            col1 = "deepskyblue"
            col2 = "blue"
            col3 = "firebrick"
            col4 = "darkorange"
            cmap = col.LinearSegmentedColormap.from_list(
                "own2", [endcolor, col1, col2, startcolor, col3, col4, endcolor]
            )

        else:
            vmin = 0
            vmid = np.log10(100)
            vmax = np.log10(10000)

            tmin = str(0)
            tmid = str(r"$10^2$")
            tmax = str(r"$10^4$")

            logscale = True
            cmap = plt.get_cmap("gist_heat")

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"

    elif tag_lookup(ame_tags, outfile):
        print("----------------------------------")
        print("Plotting AME")
        print("Applying logscale (Rewrite if not)")

        vmin = 0  # 0
        vmid = np.log10(100)
        vmax = np.log10(10000)  # 1000

        tmin = str(0)
        tmid = str(r"$10^2$")
        tmax = str(r"$10^4$")

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{ame}}$"
        logscale = True
        cmap = col.LinearSegmentedColormap.from_list("own2", ["black", "DarkOrange", "white"])

    elif tag_lookup(co10_tags, outfile):
        print("----------------------------------")
        print("Plotting CO10")
        print("Applying logscale (Rewrite if not)")
        vmin = 0
        vmid = np.log10(10)
        vmax = np.log10(100)

        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{CO10}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(co21_tags, outfile):
        print("----------------------------------")
        print("Plotting CO21")
        print("Applying logscale (Rewrite if not)")
        vmin = 0
        vmid = 1
        vmax = 2
        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{CO21}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(co32_tags, outfile):
        print("----------------------------------")
        print("Plotting 32")
        print("Applying logscale (Rewrite if not)")
        vmin = 0
        vmid = 1
        vmax = 2  # 0.5
        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{CO32}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(hcn_tags, outfile):
        print("----------------------------------")
        print("Plotting HCN")
        print("Applying logscale (Rewrite if not)")
        vmin = -14
        vmax = -10
        tmin = str(0.01)
        tmax = str(100)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + signal_labels[polt] + "$" + r"$_{\mathrm{HCN}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(ame_tags, outfile):
        print("----------------------------------")
        print("Plotting AME nu_p")

        vmin = 17
        vmax = 23
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = "GHz"
        title = r"$\nu_{ame}$"
        cmap = plt.get_cmap("bone")
        

    # SPECTRAL PARAMETER MAPS
    elif tag_lookup(dust_T_tags, outfile):
        print("----------------------------------")
        print("Plotting Thermal dust Td")

        title = r"$" + signal_labels[polt] + "$ " + r"$T_d$ "

        vmin = 14
        vmax = 30
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        cmap = plt.get_cmap("bone")


    elif tag_lookup(dust_beta_tags, outfile):
        print("----------------------------------")
        print("Plotting Thermal dust beta")

        title = r"$" + signal_labels[polt] + "$ " + r"$\beta_d$ "

        vmin = 1.3
        vmax = 1.8
        tmin = str(vmin)
        tmax = str(vmax)
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        cmap = plt.get_cmap("bone")


    elif tag_lookup(synch_beta_tags, outfile):
        print("----------------------------------")
        print("Plotting Synchrotron beta")

        title = r"$" + signal_labels[polt] + "$ " + r"$\beta_s$ "

        vmin = -4.0
        vmax = -1.5
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        cmap = plt.get_cmap("bone")


    elif tag_lookup(ff_Te_tags, outfile):
        print("----------------------------------")
        print("Plotting freefree T_e")

        vmin = 5000
        vmax = 8000
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"
        cmap = plt.get_cmap("bone")


    elif tag_lookup(ff_EM_tags, outfile):
        print("----------------------------------")
        print("Plotting freefree EM MIN AND MAX VALUES UPDATE!")

        vmin = 5000
        vmax = 8000
        tmin = str(vmin)
        tmax = str(vmax)

        vmax = np.percentile(m, 95)
        vmin = np.percentile(m, 5)
        tmin = False
        tmax = False
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"
        cmap = plt.get_cmap("bone")


    #################
    # RESIDUAL MAPS #
    #################

    elif tag_lookup(res_tags, outfile):
        print("----------------------------------")
        print("Plotting residual map" + " " + signal_labels[polt])

        if "res_" in outfile:
            tit = str(re.findall(r"res_(.*?)_", outfile)[0])
        else:
            tit = str(re.findall(r"residual_(.*?)_", outfile)[0])

        title = r"{} ".format(tit) + r"  $" + signal_labels[polt] + "$"

        vmin = -10
        vmid = 0
        vmax = 10
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)

        unit = r"$\mu\mathrm{K}$"
        cmap = col.ListedColormap(np.loadtxt(color) / 255.0)

        from pathlib import Path

        color = Path(__file__).parent / "parchment1.dat"

        if "545" in outfile:
            vmin = -1e2
            vmax = 1e2
            tmin = str(vmin)
            tmax = str(vmax)
            unit = r"$\mathrm{MJy/sr}$"
        elif "857" in outfile:
            vmin = -0.05  # -1e4
            vmax = 0.05  # 1e4
            tmin = str(vmin)
            tmax = str(vmax)
            unit = r"$\mathrm{MJy/sr}$"

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]
    #################
    # TOD MAPS      #
    #################

    elif tag_lookup(tod_tags, outfile):
        print("----------------------------------")
        print("Plotting Smap map" + " " + signal_labels[polt])

        tit = str(re.findall(r"tod_(.*?)_Smap", outfile)[0])
        title = r"{} ".format(tit) + r"  $" + signal_labels[polt] + "$"

        vmin = -0.2
        vmid = 0
        vmax = 0.2
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)

        unit = r"$\mu\mathrm{K}$"
        cmap = col.ListedColormap(np.loadtxt(color) / 255.0)

        from pathlib import Path

        color = Path(__file__).parent / "parchment1.dat"

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

    ############################
    # Not idenified or ignored #
    ############################
    elif tag_lookup(ignore_tags, outfile):
        print(
            '{} is on the ignore list, under tags {}.  Remove from "ignore_tags" in plotter.py. Breaking.'.format(
                outfile, ignore_tags
            )
        )
        sys.exit()
    else:
        print("----------------------------------")
        print("Map not recognized, plotting with min and max values")
        vmax = np.percentile(m, 95)
        vmin = np.percentile(m, 5)
        tmin = False
        tmax = False
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]
        unit = ""
        title = r"$" + signal_labels[polt] + "$"

        from pathlib import Path
        color = Path(__file__).parent / "parchment1.dat"
        cmap = col.ListedColormap(np.loadtxt(color) / 255.0)

    return title, ticks, ticklabels, unit, cmap, logscale


def get_pollist(sig):
    pollist = []
    if "I" in sig:
        pollist.append(0)
    if "Q" in sig:
        pollist.append(1)
    if "U" in sig:
        pollist.append(2)
    return pollist


def get_sizes(size):
    sizes = []
    if "s" in size:
        sizes.append(8.8)
    if "m" in size:
        sizes.append(12.0)
    if "l" in size:
        sizes.append(18.0)
    return sizes



def fmt(x, pos):
    """
    Format color bar labels
    """
    if abs(x) > 1e4:
        a, b = "{:.2e}".format(x).split("e")
        b = int(b)
        return r"${} \cdot 10^{{{}}}$".format(a, b)
    elif abs(x) > 1e2:
        return r"${:d}$".format(int(x))
    elif abs(x) > 1e1:
        return r"${:.1f}$".format(x)
    else:
        return r"${:.2f}$".format(x)


def cm2inch(cm):
    return cm * 0.393701


def tag_lookup(tags, outfile):
    return any(e in outfile for e in tags)



