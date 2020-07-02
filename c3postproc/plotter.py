import time

totaltime = time.time()
import sys
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib import rcParams, rc
from c3postproc.tools import arcmin2rad
from astropy.io import fits

print("Importtime:", (time.time() - totaltime))


def Plotter(
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
    ltitle,
    unit,
    scale,
    verbose,
):
    rcParams["backend"] = "pdf" if pdf else "Agg"
    rcParams["legend.fancybox"] = True
    rcParams["lines.linewidth"] = 2
    rcParams["savefig.dpi"] = 300
    rcParams["axes.linewidth"] = 1

    masked = False

    if darkmode:
        rcParams["text.color"] = "white"  # axes background color
        rcParams["axes.facecolor"] = "white"  # axes background color
        rcParams["axes.edgecolor"] = "white"  # axes edge color
        rcParams["axes.labelcolor"] = "white"
        rcParams["xtick.color"] = "white"  # color of the tick labels
        rcParams["ytick.color"] = "white"  # color of the tick labels
        rcParams["grid.color"] = "white"  # grid color
        rcParams[
            "legend.facecolor"
        ] = "inherit"  # legend background color (when 'inherit' uses axes.facecolor)
        rcParams[
            "legend.edgecolor"
        ] = "white"  # legend edge color (when 'inherit' uses axes.edgecolor)

    rc("text.latex", preamble=r"\usepackage{sfmath}")

    # Which signal to plot
    print("----------------------------------")
    print("Plotting " + input)

    #######################
    ####   READ MAP   #####
    #######################

    # Get maps array if .h5 file
    if input.endswith(".h5"):
        from c3postproc.commands import h5map2fits
        from c3postproc.tools import alm2fits_tool

        # Get maps from alm data in .h5
        if dataset.endswith("alm"):
            print("Converting alms to map")
            maps, nsid, lmax, fwhm, outfile = alm2fits_tool(
                input, dataset, nside, lmax, fwhm, save=False
            )

        # Get maps from map data in .h5
        elif dataset.endswith("map"):
            print("Reading map from h5")
            maps, nsid, lmax, outfile = h5map2fits(input, dataset, save=False)

        # Found no data specified kind in .h5
        else:
            print("Dataset not found. Breaking.")
            print(f"Does {input}/{dataset} exist?")
            sys.exit()

    # Plot all signals specified
    print(f"Plotting the following signals: {sig}")
    for polt in sig:
        print("----------------------------------")
        signal_label = get_signallabel(polt)

        try:
            if input.endswith(".fits"):
                map, header = hp.read_map(input, field=polt, verbose=False, h=True)
                header = dict(header)
                try:
                    signal_label = header[f"TTYPE{polt+1}"]
                except:
                    pass

                m = hp.ma(map)  # Dont use header for this
                nsid = hp.npix2nside(len(m))
                outfile = input.replace(".fits", "")

            elif input.endswith(".h5"):
                m = maps[polt]
        except:
            print(f"{polt} not found")
            sys.exit()

        ############
        #  SMOOTH  #
        ############
        if fwhm > 0 and input.endswith(".fits"):
            print(f"Smoothing fits map to {fwhm} degrees fwhm")
            m = hp.smoothing(m, fwhm=arcmin2rad(fwhm), lmax=lmax)

        ############
        # UD_GRADE #
        ############
        if nside is not None and input.endswith(".fits"):
            print(f"UDgrading map from {nsid} to {nside}")
            m = hp.ud_grade(m, nside)
        else:
            nside = nsid

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
            print(f"Dipole vector: {dip}")
            print(f"Dipole amplitude: {np.sqrt(np.sum(dip ** 2))}")

            # Create dipole template
            nside = int(nside)
            ray = range(hp.nside2npix(nside))
            vecs = hp.pix2vec(nside, ray)
            dipole = np.dot(dip, vecs)

            # Subtract dipole map from data
            m = m - dipole
            print(f"Dipole removal : {(time.time() - starttime)}") if verbose else None

        #######################
        #### Auto-param   #####
        #######################
        # Reset these every signal
        tempmin = min
        tempmax = max
        temptitle = title
        templtitle = ltitle
        tempunit = unit
        templogscale = logscale
        tempcmap = cmap

        # ttl, unt and cmb are temporary variables for title, unit and colormap
        if auto:
            ttl, lttl, ticks, unt, cmp, lgscale, scale = get_params(
                m, outfile, polt, signal_label,
            )
        else:
            ttl = ""
            unt = ""
            rng = "auto"
            ticks = [False, False]
            cmp = "planck"
            lgscale = False
        
        # Scale map
        m *= scale

        # If range has been specified, set.
        if rng:
            if rng == "auto":
                if minmax:
                    mn = np.min(m)
                    mx = np.max(m)
                else:
                    mn, mx = get_ticks(m, 97.5)
                if min is False:
                    min = mn
                if max is False:
                    max = mx
            else:
                rng = float(rng)
                min = -rng
                max = rng

        # If min and max have been specified, set.
        if min is not False:
            ticks[0] = float(min)

        if max is not False:
            ticks[-1] = float(max)

        ##########################
        #### Plotting Params #####
        ##########################

        # Upper right title
        if not title:
            title = ttl

        # Upper left title
        if not ltitle:
            ltitle = lttl
            
        # Unit under colorbar
        if not unit:
            unit = unt

        # Image size -  ratio is always 1/2
        xsize = 2000
        ysize = int(xsize / 2.0)

        ticklabels = ticks
        #######################
        ####   logscale   #####
        #######################
        # Some maps turns on logscale automatically
        # -logscale command overwrites this
        if logscale == None:
            logscale = lgscale

        if logscale:
            print("Applying logscale")
            starttime = time.time()

            m = np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))

            ticks = []
            for i in ticklabels:
                if i>0:
                    ticks.append(np.log10(i))
                elif i<0:
                    ticks.append(-np.log10(abs(i)))
                else:
                    ticks.append(i)

            m = np.maximum(np.minimum(m, ticks[-1]), ticks[0])

            print("Logscale", (time.time() - starttime)) if verbose else None

        ######################
        #### COLOR SETUP #####
        ######################
        # Chose colormap manually
        if cmap == None:
            # If not defined autoset or goto planck
            cmap = cmp

        if cmap == "planck":
            from pathlib import Path

            cmap = Path(__file__).parent / "parchment1.dat"
            cmap = col.ListedColormap(np.loadtxt(cmap) / 255.0, "planck")
        else:
            cmap = plt.get_cmap(cmap)

        print(f"Setting colormap to {cmap.name}")
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
            print(f"Masking using {mask}")
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
                longitude[::-1],
                latitude,
                grid_map,
                vmin=ticks[0],
                vmax=ticks[-1],
                rasterized=True,
                cmap=cmap,
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
                from matplotlib.ticker import FuncFormatter

                cb = fig.colorbar(
                    image,
                    orientation="horizontal",
                    shrink=0.3,
                    pad=0.08,
                    ticks=ticks,
                    format=FuncFormatter(fmt),
                )

                # Format tick labels
                print(ticklabels)
                ticklabels = [fmt(i,1) for i in ticklabels]
                cb.ax.set_xticklabels(ticklabels)
                
                cb.ax.xaxis.set_label_text(unit)
                cb.ax.xaxis.label.set_size(fontsize)
                # cb.ax.minorticks_on()

                cb.ax.tick_params(
                    which="both", axis="x", direction="in", labelsize=fontsize
                )
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

            ###################
            ## RIGHT TITLE ####
            ###################
            plt.text(
                6.0, 1.3, r"%s" % title, ha="center", va="center", fontsize=fontsize,
            )

            #############
            ## LEFT TITLE ####
            #############
            plt.text(
                -6.0, 1.3, r"%s" % ltitle, ha="center", va="center", fontsize=fontsize,
            )

            ##############
            #### SAVE ####
            ##############
            plt.tight_layout()
            filetype = "pdf" if pdf else "png"
            tp = (
                False if white_background else True
            )  # Turn on transparency unless told otherwise

            ##############
            ## filename ##
            ##############
            print(f"Using signal label {signal_label}")

            outfile = outfile.replace("_IQU_", "_")
            outfile = outfile.replace("_I_", "_")

            filename = []
            filename.append(f"{str(int(fwhm))}arcmin") if fwhm > 0 else None
            filename.append("cb") if colorbar else None
            filename.append("masked") if masked else None
            filename.append("dark") if darkmode else None
            filename.append(f"c-{cmap.name}")

            nside_tag = "_n" + str(int(nside))
            if nside_tag in outfile:
                outfile = outfile.replace(nside_tag, "")

            fn = outfile + f"_{signal_label}_w{str(int(width))}" + nside_tag

            for i in filename:
                fn += f"_{i}"
            fn += f".{filetype}"

            starttime = time.time()
            plt.savefig(
                fn,
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=tp,
                format=filetype,
            )
            print("Savefig", (time.time() - starttime)) if verbose else None

            plt.close()
            print("Totaltime:", (time.time() - totaltime)) if verbose else None

        min = tempmin
        max = tempmax
        title = temptitle
        unit = tempunit
        logscale = templogscale
        cmap = tempcmap


def get_params(m, outfile, polt, signal_label):
    print()
    logscale = False

    # Everything listed here will be recognized
    # If tag is found in output file, use template
    cmb_tags = ["cmb", "BP_cmb"]
    chisq_tags = ["chisq"]
    synch_tags = ["synch_c", "synch_amp", "BP_synch"]
    dust_tags = ["dust_c", "dust_amp", "BP_dust"]
    ame_tags = ["ame_c", "ame_amp", "ame1_c", "ame1_amp", "BP_ame"]
    ff_tags = ["ff_c", "ff_amp", "BP_freefree"]
    co10_tags = ["co10", "co-100"]
    co21_tags = ["co21", "co-217"]
    co32_tags = ["co32", "co-353"]
    hcn_tags = ["hcn"]
    ame_nup_tags = [
        "ame_nup",
        "ame_nu_p",
        "ame1_nup",
        "ame1_nu_p",
    ]
    dust_T_tags = ["dust_T", "dust_Td"]
    dust_beta_tags = ["dust_beta"]
    synch_beta_tags = ["synch_beta"]
    ff_Te_tags = ["ff_T_e", "ff_Te"]
    ff_EM_tags = ["ff_EM"]
    res_tags = ["residual_", "res_"]
    tod_tags = ["Smap"]
    freqmap_tags = ["BP_030", "BP_044", "BP_070"]
    ignore_tags = ["radio_"]

    # Simple signal label from pol index
    sl = signal_label.split("_")[0]
    ltitle = ""
    scale = 1.0 # Scale map? Default no
    startcolor = "black"
    endcolor = "white"
    cmap = "planck"  # Default cmap
    

    #######################
    # SPECTRAL INDEX MAPS #
    #######################
    if (
        tag_lookup(ame_nup_tags, outfile)
        or tag_lookup(ame_tags, outfile)
        and sidx_tag_lookup(["NU_P_MEAN"], signal_label)
    ):
        print("Plotting AME nu_p")

        ticks = [17, 23]
        unit = "GHz"
        title = r"$\nu_{peak}$"
    elif (
        tag_lookup(dust_T_tags, outfile)
        or tag_lookup(dust_tags, outfile)
        and sidx_tag_lookup(["T_MEAN"], signal_label)
    ):
        print("Plotting Thermal dust Td")

        ticks = [14, 30]
        unit = r"$\mathrm{K}$"
        title = r"$" + sl + "$ " + r"$T_d$ "
    elif (
        tag_lookup(dust_beta_tags, outfile)
        or tag_lookup(dust_tags, outfile)
        and sidx_tag_lookup(["BETA_MEAN", "BETA_P_MEAN"], signal_label)
    ):
        print("Plotting Thermal dust beta")
        ticks = [1.3, 1.8]
        unit = ""
        title = r"$" + sl + "$ " + r"$\beta_d$ "
    elif (
        tag_lookup(synch_beta_tags, outfile)
        or tag_lookup(synch_tags, outfile)
        and sidx_tag_lookup(["BETA_MEAN", "BETA_P_MEAN"], signal_label)
    ):
        print("Plotting Synchrotron beta")
        ticks = [-3.2, -3.0]
        unit = ""
        title = r"$" + sl + "$ " + r"$\beta_s$ "
    elif (
        tag_lookup(ff_Te_tags, outfile)
        or tag_lookup(ff_tags, outfile)
        and sidx_tag_lookup(["TE_MEAN"], signal_label)
    ):
        print("Plotting freefree T_e")

        ticks = [5000, 8000]
        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"

    elif tag_lookup(ff_EM_tags, outfile):
        print("Plotting freefree EM MIN AND MAX VALUES UPDATE!")

        vmin, vmax = get_ticks(m, 97.5)

        ticks = [vmin, vmax]

        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"

    #######################
    #### AMPLITUDE MAPS ###
    #######################

    elif tag_lookup(cmb_tags, outfile,):

        print(f"Plotting CMB signal {sl}")

        if polt % 3 > 0:
            ticks = [-2, 0, 2]
        else:
            ticks = [-300, 0, 300]

        unit = r"$\mu\mathrm{K}_{\mathrm{CMB}}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CMB}}$"

    elif tag_lookup(chisq_tags, outfile):

        if polt % 3 > 0:
            ticks = [0, 32]
        else:
            ticks = [0, 76]

        print("Plotting chisq with vmax = " + str(ticks[-1]) + " " + sl)

        unit = ""
        title = r"$\chi^2$ " + sl
        cmap = col.LinearSegmentedColormap.from_list("BkWh", ["black", "white"])

    elif tag_lookup(synch_tags, outfile):
        print(f"Plotting Synchrotron {sl}")
        title = r"$" + sl + "$" + r"$_{\mathrm{synch.}}$ "
        if polt % 3 > 0:
            # BP uses 30 GHz ref freq for pol

            ticks = [-50, 0, 50]
            logscale = True
            unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        else:
            # BP uses 408 MHz GHz ref freq
            #scale = 1e-6
            ticks = [10, 30, 100, 300]
            logscale = True
            unit = r"$\mathrm{K}_{\mathrm{RJ}}$"

    elif tag_lookup(ff_tags, outfile):
        print("Plotting freefree")

        ticks = [0, 100, 10000]
        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{free-free}}$"
        logscale = True

    elif tag_lookup(dust_tags, outfile):
        print("Plotting Thermal dust" + " " + sl)

        if polt % 3 > 0:
            ticks = [-100, 0, 100]
            logscale = True

            col1 = "deepskyblue"
            col2 = "blue"
            col3 = "firebrick"
            col4 = "darkorange"
            cmap = col.LinearSegmentedColormap.from_list(
                "BlBkRd", [endcolor, col1, col2, startcolor, col3, col4, endcolor],
            )

        else:
            ticks = [0, 100, 10000]
            logscale = True
            cmap = plt.get_cmap("gist_heat")

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{d}}$ "

    elif tag_lookup(ame_tags, outfile):
        print("Plotting AME")

        ticks = [0, 100, 10000]
        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{ame}}$"
        logscale = True

    #######################
    ## LINE EMISSION MAPS #
    #######################

    elif tag_lookup(co10_tags, outfile):
        print("Plotting CO10")

        ticks = [0, 10, 100]
        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CO10}}$"
        logscale = True

    elif tag_lookup(co21_tags, outfile):
        print("Plotting CO21")

        ticks = [0, 10, 100]
        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CO21}}$"
        logscale = True

    elif tag_lookup(co32_tags, outfile):
        print("Plotting 32")

        ticks = [0, 10, 100]
        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CO32}}$"
        logscale = True

    elif tag_lookup(hcn_tags, outfile):
        print("Plotting HCN")

        ticks = [0.01, 100]
        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{HCN}}$"
        logscale = True

    #################
    # RESIDUAL MAPS #
    #################

    elif tag_lookup(res_tags, outfile):
        from re import findall

        print("Plotting residual map" + " " + sl)

        if "res_" in outfile:
            tit = str(findall(r"res_(.*?)_", outfile)[0])
        else:
            tit = str(findall(r"residual_(.*?)_", outfile)[0])

        title = fr"{tit} " + r"  $" + sl + "$"

        ticks = [-10, 0, 10]
        unit = r"$\mu\mathrm{K}$"

        if "545" in outfile:
            ticks = [-100, 0, 100]
            unit = r"$\mathrm{MJy/sr}$"
        elif "857" in outfile:
            ticks = [-0.05, 0, 0.05]
            unit = r"$\mathrm{MJy/sr}$"

    ############
    # TOD MAPS #
    ############

    elif tag_lookup(tod_tags, outfile):
        from re import findall

        print("Plotting Smap map" + " " + sl)

        tit = str(findall(r"tod_(.*?)_Smap", outfile)[0])
        title = fr"{tit} " + r"  $" + sl + "$"

        ticks = [-0.2, 0, 0.2]
        unit = r"$\mu\mathrm{K}$"

    ############
    # FREQMAPS #
    ############

    elif tag_lookup(freqmap_tags, outfile):
        from re import findall

        print("Plotting Frequency map" + " " + sl)

        vmin, vmax = get_ticks(m, 97.5)
        ticks = [vmin, 0.0, vmax]

        unit = r"$\mu\mathrm{K}$"
        tit = str(findall(r"BP_(.*?)_", outfile)[0])
        tl =  sl + "_{"+fr"{tit}"+"}"
        title = r"$" + tl + "$"
        
    ############################
    # Not idenified or ignored #
    ############################
    elif tag_lookup(ignore_tags, outfile):
        print(
            f'{outfile} is on the ignore list, under tags {ignore_tags}. Remove from "ignore_tags" in plotter.py. Breaking.'
        )
        sys.exit()
    else:
        # Run map autoset
        return not_identified(m, signal_label, logscale,)

    # If signal is an RMS map, add tag.
    if signal_label.endswith("RMS"):
        print(f"Plotting RMS signal {signal_label}")
        ltitle = "RMS"
        vmin, vmax = get_ticks(m, 97.5)
        logscale=False
        ticks = [vmin, vmax]        
        unit = ""
        cmap = "planck"
        scale = 1.0

    return title, ltitle, ticks, unit, cmap, logscale, scale


def not_identified(m, signal_label, logscale):
    print("Map not recognized, plotting with min and max values")
    scale = 1.0
    vmin, vmax = get_ticks(m, 97.5)
    ticks = [vmin, vmax]

    unit = ""
    tl = signal_label.split("_")[0] + "_{" + signal_label.split("_")[-1] + "}"
    title = r"$" + tl + "$"
    ltitle = ""
    cmap = "planck"

    return title, ltitle, ticks, unit, cmap, logscale, scale


def get_signallabel(x):
    if x == 0:
        return "I"
    if x == 1:
        return "Q"
    if x == 2:
        return "U"
    return str(x)


def get_sizes(size):
    sizes = []
    if "s" in size:
        sizes.append(8.8)
    if "m" in size:
        sizes.append(12.0)
    if "l" in size:
        sizes.append(18.0)
    return sizes

def get_ticks(m, percentile):
    vmin = np.percentile(m, 100.0-percentile)
    vmax = np.percentile(m, percentile)

    vmin = 0.0 if abs(vmin) < 1e-5 else vmin
    vmax = 0.0 if abs(vmax) < 1e-5 else vmax
    return vmin, vmax


def fmt(x, pos):
    """
    Format color bar labels
    """

    if abs(x) > 1e4 or (abs(x) < 1e-2 and abs(x) > 0):
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        return fr"${a} \cdot 10^{{{b}}}$"
    elif abs(x) > 1e2 or (float(abs(x))).is_integer():
        return fr"${int(x):d}$"
    elif abs(x) > 1e1:
        return fr"${x:.1f}$"
    elif abs(x) == 0.0:
        return fr"${x:.1f}$"
    else:
        return fr"${x:.2f}$"

def cm2inch(cm):
    return cm * 0.393701


def sidx_tag_lookup(tags, signal_label):
    return any(e in signal_label for e in tags)


def tag_lookup(tags, outfile):
    return any(e in outfile for e in tags)
