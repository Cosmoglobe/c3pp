import time

totaltime = time.time()
import click
import sys
import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib import rcParams, rc
from src.tools import arcmin2rad
from astropy.io import fits

print("Importtime:", (time.time() - totaltime))


def Plotter(input, dataset, nside, auto, min, max, mid, rng, colorbar, lmax, fwhm, mask, mfill, sig, remove_dipole, remove_monopole, logscale, size, white_background, darkmode, png, cmap, title, ltitle, unit, scale, outdir, verbose, data, labelsize):
    rcParams["backend"] = "agg" if png else "pdf"
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
        rcParams["legend.facecolor"] = "inherit"  # legend background color (when 'inherit' uses axes.facecolor)
        rcParams["legend.edgecolor"] = "white"  # legend edge color (when 'inherit' uses axes.edgecolor)

    rc("text.latex", preamble=r"\usepackage{sfmath}",)

    # Which signal to plot
    click.echo("")
    click.echo(click.style("{:#^48}".format(""), fg="green"))
    click.echo(click.style("Plotting",fg="green") + f" {input}")

    #######################
    ####   READ MAP   #####
    #######################

    # Get maps array if .h5 file
    if input.endswith(".h5"):
        from src.commands_hdf import h5map2fits
        from src.tools import alm2fits_tool

        # Get maps from alm data in .h5
        if dataset.endswith("alm"):
            click.echo(click.style("Converting alms to map",fg="green"))
            (maps, nsid, lmax, fwhm, outfile,) = alm2fits_tool(input, dataset, nside, lmax, fwhm, save=False,)

        # Get maps from map data in .h5
        elif dataset.endswith("map"):
            click.echo(click.style("Reading map from hdf",fg="green"))
            (maps, nsid, lmax, outfile,) = h5map2fits(input, dataset, save=False)

        # Found no data specified kind in .h5
        else:
            click.echo(click.style("Dataset not found. Breaking.",fg="red"))
            click.echo(click.style(f"Does {input}/{dataset} exist?",fg="red"))
            sys.exit()

    # Plot all signals specified
    click.echo(click.style("Using signals ",fg="green") + f"{sig}")
    click.echo(click.style("{:#^48}".format(""), fg="green"))
    for polt in sig:
        signal_label = get_signallabel(polt)
        if data is not None:
            m = data.copy()
            m = hp.ma(m)
            nsid = hp.get_nside(m)
            outfile = input.replace(".fits", "")
        else:
            try:
                if input.endswith(".fits"):
                    map, header = hp.read_map(input, field=polt, verbose=False, h=True, dtype=None,)
                    header = dict(header)
                    try:
                        signal_label = header[f"TTYPE{polt+1}"]
                        if signal_label in ["TEMPERATURE", "TEMP"]:
                            signal_label = "T"
                        if signal_label in ["Q-POLARISATION", "Q_POLARISATION"]:
                            signal_label = "Q"
                        if signal_label in ["U-POLARISATION", "U_POLARISATION"]:
                            signal_label = "U"
                        
                    except:
                        pass
            
                    m = hp.ma(map)  # Dont use header for this
                    nsid = hp.get_nside(m)
                    outfile = input.replace(".fits", "")
            
                elif input.endswith(".h5"):
                    m = maps[polt]
            except:
                click.echo(click.style(f"{polt} not found",fg="red"))
                sys.exit()
        
        ############
        #  SMOOTH  #
        ############
        if float(fwhm) > 0 and input.endswith(".fits"):
            click.echo(click.style(f"Smoothing fits map to {fwhm} arcmin fwhm",fg="yellow"))
            m = hp.smoothing(m, fwhm=arcmin2rad(fwhm), lmax=lmax,)

        ############
        # UD_GRADE #
        ############
        if nside is not None and input.endswith(".fits"):
            if nsid != nside:
                click.echo(click.style(f"UDgrading map from {nsid} to {nside}", fg="yellow"))
                m = hp.ud_grade(m, nside,)
        else:
            nside = nsid

        ########################
        #### remove dipole #####
        ########################
        if remove_dipole or remove_monopole:
            starttime = time.time()

            if remove_monopole:
                dip_mask_name = remove_monopole
            if remove_dipole:
                dip_mask_name = remove_dipole
            # Mask map for dipole estimation
            if dip_mask_name == 'auto':
                mono, dip = hp.fit_dipole(m, gal_cut=30)
            else:
                m_masked = hp.ma(m)
                m_masked.mask = np.logical_not(hp.read_map(dip_mask_name,verbose=False,dtype=None,))

                # Fit dipole to masked map
                mono, dip = hp.fit_dipole(m_masked)

            # Subtract dipole map from data
            if remove_dipole:
                click.echo(click.style("Removing dipole:", fg="yellow"))
                click.echo(click.style("Dipole vector:",fg="green") + f" {dip}")
                click.echo(click.style("Dipole amplitude:",fg="green") + f" {np.sqrt(np.sum(dip ** 2))}")

                # Create dipole template
                nside = int(nside)
                ray = range(hp.nside2npix(nside))
                vecs = hp.pix2vec(nside, ray)
                dipole = np.dot(dip, vecs)
                
                m = m - dipole
            if remove_monopole:
                click.echo(click.style("Removing monopole:", fg="yellow"))
                click.echo(click.style("Mono:",fg="green") + f" {mono}")
                m = m - mono
            
            click.echo(f"Dipole removal : {(time.time() - starttime)}") if verbose else None

        #######################
        #### Auto-param   #####
        #######################
        # Reset these every signal
        tempmin = min
        tempmax = max
        tempmid = mid
        temptitle = title
        templtitle = ltitle
        tempunit = unit
        templogscale = logscale
        tempcmap = cmap

        # Scale map
        if scale:
            click.echo(click.style(f"Scaling data by {scale}",fg="yellow"))
            m *= scale

        if auto:
            (_title, ticks, cmp, lgscale,) = get_params(m, outfile, polt, signal_label,)
            # Title
            if _title["stddev"]:
                if _title["special"]:
                    ttl =   r"$\sigma_{\mathrm{" + _title["param"].replace("$","") + "}}$"
                else:
                    #_title["param"] + r"$_{\mathrm{" + _title["comp"] + "}}^{\sigma}$" 
                    ttl =   r"$\sigma_{\mathrm{" + _title["comp"] + "}}$"
            elif _title["rms"]:
                ttl =  _title["param"] + r"$_{\mathrm{" + _title["comp"] + "}}^{\mathrm{RMS}}$" 
            elif _title["mean"]:
                #r"$\langle$" + _title["param"] + r"$\rangle$" + r"$_{\mathrm{" + _title["comp"] + "}}^{ }$" 
                ttl = r"$\langle$" + _title["param"] + r"$_{\mathrm{" + _title["comp"] + "}}^{ }$" + r"$\rangle$"  
            elif _title["diff"]:
                ttl = r"$\Delta$ " + _title["param"] + r"$_{\mathrm{" + _title["comp"] + "}}^{" + _title["diff_label"] + "}$" 
            else:
                ttl =  _title["param"] + r"$_{\mathrm{" + _title["comp"] + "}}^{ }$" 
            try:
                ttl = _title["custom"]
            except:
                pass

            # Left signal label
            lttl = r"$" + _title["sig"] +"$"
            if lttl == "$I$":
                lttl = "$T$"
            elif lttl == "$QU$":
                lttl= "$P$"

            # Tick bug fix
            mn = ticks[0]
            mx = ticks[-1]
            md = None
            if not mid and len(ticks)>2:
                if ticks[0]<ticks[1]  and ticks[-2]<ticks[-1]:
                    md = ticks[1:-1]
                else:
                    ticks.pop(1)


            # Unit
            unt = _title["unit"]
        else:
            ttl = ""
            lttl = ""
            unt = ""
            md = None
            ticks = [False, False]
            cmp = "planck"
            lgscale = False


        # If min and max have been specified, set.
        if rng == "auto" and not auto:
            click.echo(click.style("Setting range from 97.5th percentile of data",fg="yellow"))
            mn, mx = get_ticks(m, 97.5)
        elif rng == "minmax":
            click.echo(click.style("Setting range from min to max of data",fg="yellow"))
            mn = np.min(m)
            mx = np.max(m)
        else:
            try:
                if float(rng)>0.0:
                    mn = -float(rng)
                    mx = float(rng)
                    ticks = [False, 0.0, False]
            except:
                pass

        if min is False:
            min = mn
        else:
            min = float(min)

        if max is False:
            max = mx
        else:
            max = float(max)   

        ticks[0] = min
        ticks[-1] = max
        if mid:
            ticks = [min, *mid, max]
        elif md:
            ticks = [min, *md, max] 
        ticks = [float(i) for i in ticks]
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
            click.echo(click.style("Applying semi-logscale", fg="yellow", blink=True, bold=True))
            starttime = time.time()
            linthresh=1
            m = symlog(m,linthresh)

            ticks = []
            for i in ticklabels:
                ticks.append(symlog(i,linthresh))

            m = np.maximum(np.minimum(m, ticks[-1]), ticks[0])

            click.echo("Logscale", (time.time() - starttime),) if verbose else None

        ######################
        #### COLOR SETUP #####
        ######################
        # Chose colormap manually
        if cmap == None:
            # If not defined autoset or goto planck
            cmap = cmp
        if cmap == "planck":
            from pathlib import Path
            if False: #logscale:
                cmap = Path(__file__).parent / "planck_cmap_logscale.dat"
            else:
                cmap = Path(__file__).parent / "planck_cmap.dat"
            cmap = col.ListedColormap(np.loadtxt(cmap) / 255.0, "planck")
        elif cmap.startswith("q-"):
            import plotly.colors as pcol
            _, clab, *numvals = cmap.split("-")
            colors = getattr(pcol.qualitative, clab)
            if clab=="Plotly":
                #colors.insert(3,colors.pop(-1))
                colors.insert(0,colors.pop(-1))
                colors.insert(3,colors.pop(2))
            try:
                cmap = col.ListedColormap(colors[:int(numvals[0])], f'{clab}-{numvals[0]}')
                click.echo(click.style("Using qualitative colormap:", fg="yellow") + f" {clab} up to {numvals[0]}")
            except:
                cmap = col.ListedColormap(colors,clab)
                click.echo(click.style("Using qualitative colormap:", fg="yellow") + f" {clab}")
        elif cmap.startswith("black2"):
            cmap = col.LinearSegmentedColormap.from_list(cmap,cmap.split("2"))
        else:
            try:
                import cmasher
                cmap = eval(f"cmasher.{cmap}")
            except:
                cmap = plt.get_cmap(cmap)

        click.echo(click.style("Colormap:", fg="green") + f" {cmap.name}")
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
            click.echo(click.style(f"Masking using {mask}", fg="yellow"))
            masked = True

            # Apply mask
            hp.ma(m)
            mask_field = polt-3 if polt>2 else polt
            m.mask = np.logical_not(hp.read_map(mask, field=mask_field, verbose=False, dtype=None))

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

        # Format tick labels
        click.echo(click.style("Ticks: ", fg="green") + f"{ticklabels}")
        ticklabels = [fmt(i, 1) for i in ticklabels]
        click.echo(click.style("Unit: ", fg="green") + f"{unit}")
        click.echo(click.style("Title: ", fg="green") + f"{title}")


        sizes = get_sizes(size)
        for width in sizes:
            # Size of plot
            click.echo(click.style("Size: ", fg="green") + str(width))
            height = width / 2.0
            height *= 1.275 if colorbar else 1 # Make sure text doesnt change with colorbar

            ################
            ##### font #####
            ################
            fontsize = 10

            fig = plt.figure(figsize=(cm2inch(width), cm2inch(height),),)
            ax = fig.add_subplot(111, projection="mollweide")

            # rasterized makes the map bitmap while the labels remain vectorial
            # flip longitude to the astro convention
            image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=ticks[0], vmax=ticks[-1], rasterized=True, cmap=cmap, shading='auto',)
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
                from matplotlib.ticker import FuncFormatter, LogLocator
                cb = fig.colorbar(image, orientation="horizontal", shrink=0.4, pad=0.08, ticks=ticks, format=FuncFormatter(fmt),)

                cb.ax.set_xticklabels(ticklabels)
                cb.ax.xaxis.set_label_text(unit)
                cb.ax.xaxis.label.set_size(fontsize)
                if logscale:
                    #if f == 0:
                    #    linticks = np.array([])
                    #else:
                    linticks = np.linspace(-1, 1, 3)*linthresh
                    logmin = np.round(ticks[0])
                    logmax = np.round(ticks[-1])

                    logticks_min = -10**np.arange(0, abs(logmin)+1)
                    logticks_max = 10**np.arange(0, logmax+1)
                    ticks_ = np.unique(np.concatenate((logticks_min, linticks, logticks_max)))
                    #cb.set_ticks(np.concatenate((ticks,symlog(ticks_))), []) # Set major ticks

                    logticks = symlog(ticks_, linthresh)
                    logticks = [x for x in logticks if x not in ticks]
                    cb.set_ticks(np.concatenate((ticks,logticks ))) # Set major ticks
                    cb.ax.set_xticklabels(ticklabels + ['']*len(logticks))

                    minorticks = np.linspace(-linthresh, linthresh, 5)
                    minorticks2 = np.arange(2,10)*linthresh

                    for i in range(len(logticks_min)):
                        minorticks = np.concatenate((-10**i*minorticks2,minorticks))
                    for i in range(len(logticks_max)):
                        minorticks = np.concatenate((minorticks, 10**i*minorticks2))

                    minorticks = symlog(minorticks, linthresh)
                    minorticks = minorticks[ (minorticks >= ticks[0]) & ( minorticks<= ticks[-1]) ] 
                    cb.ax.xaxis.set_ticks(minorticks, minor=True)

                

                cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize,)
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
            plt.text(4.5, 1.1, r"%s" % title, ha="center", va="center", fontsize=labelsize,)

            ##################
            ## LEFT TITLE ####
            ##################
            plt.text(-4.5, 1.1, r"%s" % ltitle, ha="center", va="center", fontsize=labelsize,)

            ##############
            #### SAVE ####
            ##############
            plt.tight_layout()
            filetype = "png" if png else "pdf"
            # Turn on transparency unless told otherwise
            tp = False if white_background else True  

            ##############
            ## filename ##
            ##############
            outfile = outfile.replace("_IQU_", "_")
            outfile = outfile.replace("_I_", "_")

            filename = []
            filename.append(f"{str(int(fwhm))}arcmin") if float(fwhm) > 0 else None
            filename.append("cb") if colorbar else None
            filename.append("masked") if masked else None
            filename.append("nodip") if remove_dipole else None
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
            if outdir:
                fn = outdir + "/" + os.path.split(fn)[-1]

            click.echo(click.style("Output:", fg="green") + f" {fn}")
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.02, transparent=tp, format=filetype,)
            click.echo("Savefig", (time.time() - starttime),) if verbose else None

            plt.close()
            click.echo("Totaltime:", (time.time() - totaltime),) if verbose else None

        min = tempmin
        max = tempmax
        mid = tempmid
        mn = mx = md = None
        title = temptitle
        ltitle = temptitle
        unit = tempunit
        logscale = templogscale
        cmap = tempcmap


def get_params(m, outfile, polt, signal_label):
    outfile = os.path.split(outfile)[-1] # Remove path 
    logscale = False

    # Everything listed here will be recognized
    # If tag is found in output file, use template
    cmb_tags = ["cmb", "BP_cmb"]
    chisq_tags = ["chisq"]
    synch_tags = ["synch",]
    dust_tags = ["dust",]
    ame_tags = [ "ame_","ame1","ame2",]
    ff_tags = ["ff_", "freefree",]
    co10_tags = ["co10", "co-100"]
    co21_tags = ["co21", "co-217"]
    co32_tags = ["co32", "co-353"]
    hcn_tags = ["hcn"]
    ame_nup_tags = ["_nup","_nu_p", "_NU_P",]
    dust_T_tags = ["_T_", "Td"]
    dust_beta_tags = ["beta","BETA",]
    synch_beta_tags = ["beta", "BETA",]
    ff_Te_tags = ["T_e", "_Te", "_TE_", "_T_E",]
    ff_EM_tags = ["_EM"]
    res_tags = ["residual_", "res_"]
    tod_tags = ["Smap"]
    freqmap_tags = ["BP_030", "BP_044", "BP_070"]
    bpcorr_tags = ["_bpcorr_"]
    ignore_tags = ["radio_"]
    mask_tags = ["mask"]

    # Simple signal label from pol index
    title = {}
    sl = signal_label.split("_")[0]
    title["sig"] = sl
    title["unit"] = ""
    title["special"] = False
    startcolor = "black"
    endcolor = "white"
    cmap = "planck"  # Default cmap

    #######################
    #### Component maps ###
    #######################

    # ------ CMB ------
    if tag_lookup(cmb_tags, outfile,):
        click.echo(click.style("{:-^48}".format(f"Detected CMB signal {sl}"),fg="yellow"))
        title["unit"]  = r"$\mu\mathrm{K}_{\mathrm{CMB}}$"
        title["comp"]  = "cmb" 
        title["param"] = r"$A$"
        if sl == "Q" or sl == "U" or sl == "QU" or sl=="P":
            ticks = [-10, 0, 10]
        else:
            ticks = [-300, 0, 300]
    # ------ Chisq ------
    elif tag_lookup(chisq_tags, outfile):
        click.echo(click.style("{:-^48}".format(f"Detected chisq {sl}"),fg="yellow"))
        title["special"] = True
        title["unit"]  = ""
        title["comp"]  = ""
        title["param"] = r"$\chi^2$"
        vmin, vmax = get_ticks(m, 97.5)
        ticks = [vmin, vmax]
        cmap = "binary"

    # ------ SYNCH ------
    elif tag_lookup(synch_tags, outfile):
        title["comp"] = "s"
        if tag_lookup(synch_beta_tags, outfile+signal_label,):
            click.echo(click.style("{:-^48}".format(f"Detected Synchrotron beta"),fg="yellow"))
            click.echo("Detected Synchrotron beta")
            ticks = [-3.2, -3.1, -3.0]
            title["unit"]  = ""
            title["param"] = r"$\beta$"
        else:
            click.echo(click.style("{:-^48}".format(f"Detected Synchrotron {sl}"),fg="yellow"))            
            title["param"] = r"$A$"
            logscale = True
            cmap = "swamp"
            if sl == "Q" or sl == "U" or sl == "QU":
                # BP uses 30 GHz ref freq for pol
                title["unit"] = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
                cmap = "wildfire"
                ticks = [-100, 0, 100]
            elif title["sig"] == "P": 
                title["unit"] = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
                ticks = [5, 10, 25, 50]
            else:
                # scale = 1e-6
                ticks = [50, 100, 200, 400]
                title["unit"] = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"

    # ------ FREE-FREE ------
    elif tag_lookup(ff_tags, outfile) and not tag_lookup(["diff",], outfile+signal_label,):
        title["comp"]  = "ff" 
        if tag_lookup(ff_Te_tags, outfile+signal_label,):
            click.echo(click.style("{:-^48}".format(f"Detected free-free Te"),fg="yellow"))
            ticks = [5000, 8000]
            title["unit"]  = r"$\mathrm{K}$"
            title["param"] = r"$T_{e}$"
        elif tag_lookup(ff_EM_tags, outfile+signal_label,):
            click.echo("Detected freefree EM MIN AND MAX VALUES UPDATE!")
            vmin, vmax = get_ticks(m, 97.5)
            ticks = [vmin, vmax]
            title["unit"]  = r"$\mathrm{K}$"
            title["param"] = r"$T_{e}$"
        else:
            click.echo(click.style("{:-^48}".format(f"Detected free-free"),fg="yellow"))
            title["param"] = r"$A$"
            cmap = "freeze"
            ticks = [40, 400, 2000]
            title["unit"] = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
            logscale = True

    # ------ AME ------
    elif tag_lookup(ame_tags, outfile):
        title["comp"]  = "ame"
        if tag_lookup(ame_nup_tags, outfile+signal_label,):
            click.echo(click.style("{:-^48}".format(f"Detected AME nu_p"),fg="yellow"))        
            ticks = [17, 23]
            title["unit"]  = "GHz"
            title["param"] = r"$\nu_{p}$"
        else:
            click.echo(click.style("{:-^48}".format(f"Detected AME"),fg="yellow"))
            title["param"] = r"$A$"
            title["unit"] = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
            cmap = "amber"
            ticks = [30, 300, 3000]
            logscale = True

    # ------ THERMAL DUST ------
    elif tag_lookup(dust_tags, outfile):
        title["comp"]  = "d"
        if tag_lookup(dust_T_tags, outfile+signal_label,):
            click.echo(click.style("{:-^48}".format(f"Detected Thermal dust temperature"),fg="yellow"))
            ticks = [14, 30]
            title["unit"]  = r"$\mathrm{K}$"
            title["param"] = r"$T$"
        elif tag_lookup(dust_beta_tags, outfile+signal_label,):
            click.echo(click.style("{:-^48}".format(f"Detected Thermal dust beta"),fg="yellow"))
            ticks = [1.3, 1.8]
            title["unit"]  = ""
            title["param"] = r"$\beta$"
        else:
            click.echo(click.style("{:-^48}".format(f"Detected Thermal dust {sl}"),fg="yellow"))
            title["param"] = r"$A$"
            title["unit"] = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
            logscale = True

            if sl == "Q" or sl == "U" or sl == "QU":
                ticks = [-100, 0, 100]
                cmap = "iceburn"
            elif sl=="P":
                cmap = "sunburst"
                ticks = [5, 10, 25, 50] #10, 100, 1000]
            else:
                cmap = "sunburst"
                ticks = [30, 300, 3000]
                

    #######################
    ## LINE EMISSION MAPS #
    #######################

    elif tag_lookup(co10_tags, outfile):
        click.echo(click.style("{:-^48}".format(f"Detected CO10"),fg="yellow"))
        click.echo("Detected CO10")
        cmap = "arctic"
        title["comp"] = "co10"
        title["param"] = r"$A$"
        title["unit"] = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        
        ticks = [0, 10, 100]

        logscale = True

    elif tag_lookup(co21_tags, outfile):
        click.echo(click.style("{:-^48}".format(f"Detected CO21"),fg="yellow"))
        title["comp"] = "co21"
        title["param"] =r"$A$"
        title["unit"] = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        ticks = [0, 10, 100]
        cmap = "arctic"
        logscale = True

    elif tag_lookup(co32_tags, outfile):
        click.echo(click.style("{:-^48}".format(f"Detected CO32"),fg="yellow"))
        title["comp"] = "co32"
        title["param"] = r"$A$"
        ticks = [0, 10, 100]
        cmap = "arctic"
        title["unit"] = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        logscale = True

    elif tag_lookup(hcn_tags, outfile):
        click.echo(click.style("{:-^48}".format(f"Detected HCN"),fg="yellow"))
        title["comp"] = "hcn"
        title["param"] = r"$A$"
        ticks = [0.01, 100]
        cmap = "arctic"
        title["unit"] = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        logscale = True

    #################
    # RESIDUAL MAPS #
    #################

    elif tag_lookup(res_tags, outfile):
        from re import findall
        click.echo(click.style("{:-^48}".format(f"Detected residual map {sl}"),fg="yellow"))

        if "res_" in outfile:
            if "WMAP" in outfile:
                tit = str(findall(r"WMAP_(.*?)_", outfile)[0])
            else:
                tit = str(findall(r"res_(.*?)_", outfile)[0])
        else:
            tit = str(findall(r"residual_(.*?)_", outfile)[0])
        
        ticks = [-3, 0, 3]
        
        if "WMAP" in outfile:
            if "_P_" in outfile and sl != "T":
                ticks = [-10, 0, 10]
            
        title["comp"] = fr"{tit}"
        title["param"] = r"$r$"
        title["unit"] = r"$\mu\mathrm{K}$"

        if "545" in outfile:
            ticks = [-100, 0, 100]
            title["unit"] = r"$\mathrm{MJy/sr}$"
        elif "857" in outfile:
            ticks = [-0.05, 0, 0.05]
            title["unit"] = r"$\mathrm{MJy/sr}$"

    ############
    # TOD MAPS #
    ############
    elif tag_lookup(tod_tags, outfile):
        from re import findall
        click.echo(click.style("{:-^48}".format(f"Detected Smap {sl}"),fg="yellow"))
        
        tit = str(findall(r"tod_(.*?)_Smap", outfile)[0])
        title["comp"] = fr"{tit}"
        title["param"] = r"$smap$"
        ticks = [-0.2, 0, 0.2]
        title["unit"] = r"$\mu\mathrm{K}$"

    ############
    # FREQMAPS #
    ############

    elif tag_lookup(freqmap_tags, outfile):
        from re import findall
        click.echo(click.style("{:-^48}".format(f"Detected frequency map {sl}"),fg="yellow"))
        
        vmin, vmax = get_ticks(m, 97.5)
        ticks = [vmin, vmax]
        #logscale=True
        #if logscale:
        #    ticks = [-1e3, 0, 1e3, 1e7]
        title["unit"] = r"$\mu\mathrm{K}$"
        tit = str(findall(r"BP_(.*?)_", outfile)[0])
        title["comp"] = fr"{tit}"
        title["param"] = r"$A$"

    ############
    #  BPcorr  #
    ############

    elif tag_lookup(bpcorr_tags, outfile):
        from re import findall
        click.echo(click.style("{:-^48}".format(f"Detected BPcorr map {sl}"),fg="yellow"))
        
        ticks = [-30,0,30]
        title["unit"] = r"$\mu\mathrm{K}$"
        tit = str(findall(r"tod_(.*?)_bpcorr", outfile)[0])

        title["comp"] = fr"{tit}"
        title["param"] = r"$s$"
        title["custom"] = title["param"] + r"$_{\mathrm{leak}}^{" + title["comp"].lstrip('0') + "}}$"

    #############
    #  Mask     #
    #############

    elif tag_lookup(mask_tags, outfile):
        from re import findall
        click.echo(click.style("{:-^48}".format(f"Detected mask file {sl}"),fg="yellow"))
        
        ticks = [0, 1]
        title["unit"] = r""

        title["comp"] = r"Mask"
        title["param"] = r""
        title["custom"] = title["Comp"]

        cmap = "bone"
 

    ############################
    # Not idenified or ignored #
    ############################
    elif tag_lookup(ignore_tags, outfile):
        click.echo(f'{outfile} is on the ignore list, under tags {ignore_tags}. Remove from "ignore_tags" in plotter.py. Breaking.')
        sys.exit()
    else:
        # Run map autoset
        title, ticks, cmap, logscale, = not_identified(m, signal_label, logscale, title)

    # If signal is an RMS map, add tag.
    if signal_label.endswith("STDDEV") or "_stddev" in outfile:
        click.echo(click.style(f"Detected STDDEV map",fg="yellow"))
        title["stddev"] = True
        vmin, vmax = get_ticks(m, 97.5)
        logscale = False
        #vmin = 0
        ticks = [vmin, vmax]
        cmap = "planck"
    else:
        title["stddev"] = False

    # If signal is an RMS map, add tag.
    if signal_label.endswith("RMS") or "_rms" in outfile:
        click.echo(click.style(f"Detected RMS map",fg="yellow"))
        title["rms"] = True
        vmin, vmax = get_ticks(m, 97.5)
        logscale = False
        #vmin = 0
        ticks = [vmin, vmax]
        cmap = "planck"
    else:
        title["rms"] = False

    if tag_lookup(["diff"], outfile):
        if tag_lookup(["dx12"], outfile):
            title["diff_label"] = "\mathrm{2018}"
        elif tag_lookup(["npipe"], outfile):
            title["diff_label"] = "\mathrm{NPIPE}"
        else:
            title["diff_label"] = ""

        vmin, vmax = get_ticks(m, 97.5)
        logscale = False
        #vmin = 0
        ticks = [vmin, vmax]
        title["diff"] = True 
    else:
        title["diff"] = False

    if signal_label.endswith("MEAN") or "_mean" in outfile:
        click.echo(click.style(f"Detected MEAN map",fg="yellow"))
        title["mean"] = True 
    else:
        title["mean"] = False

    title["comp"] = title["comp"].lstrip('0')    

    return (title, ticks, cmap, logscale,)


def not_identified(m, signal_label, logscale, title):
    click.echo(click.style("{:-^48}".format(f"Map not recognized, plotting with min and max values"),fg="yellow"))
    title["comp"] = signal_label.split("_")[-1]
    title["param"] = ""
    title["unit"] = ""
    vmin, vmax = get_ticks(m, 97.5)
    ticks = [vmin, vmax]
    cmap = "planck"
    return (title, ticks, cmap, logscale,)


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
    vmin = np.percentile(m, 100.0 - percentile)
    vmax = np.percentile(m, percentile)

    mag = max(abs(vmin), abs(vmax))
    vmin = np.sign(vmin)*mag
    vmax = np.sign(vmax)*mag

    vmin = 0.0 if abs(vmin) < 1e-5 else vmin
    vmax = 0.0 if abs(vmax) < 1e-5 else vmax


    return vmin, vmax


def fmt(x, pos):
    """
    Format color bar labels
    """
    if abs(x) > 1e4 or (abs(x) <= 1e-3 and abs(x) > 0):
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        if float(a) == 1.00:
            return r"$10^{"+str(b)+"}$"
        elif float(a) == -1.00:
            return r"$-10^{"+str(b)+"}$"
        else:
            return fr"${a} \cdot 10^{b}$"
    elif abs(x) > 1e1 or (float(abs(x))).is_integer():
        return fr"${int(x):d}$"
    else:
        x = round(x, 2)
        return fr"${x}$"


def cm2inch(cm):
    return cm * 0.393701

def tag_lookup(tags, outfile):
    return any(e in outfile for e in tags)

def symlog(m, linthresh=1.0):
    # Extra fact of 2 ln 10 makes symlog(m) = m in linear regime
    m = m/linthresh/(2*np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))
