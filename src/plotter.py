import time
totaltime = time.time()
import click
import sys
import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from src.tools import arcmin2rad

print("Importtime:", (time.time() - totaltime))

def Plotter(input, dataset, nside, auto, min, max, mid, rng, colorbar,
            graticule, lmax, fwhm, mask, mfill, sig, remove_dipole, remove_monopole,
            logscale, size, white_background, darkmode, png, cmap, title,
            ltitle, unit, scale, outdir, verbose, data, labelsize, gif, oldfont):
    fontsize = 11
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = 'Times'
    if not oldfont:
        plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["backend"] = "agg" if png else "pdf"
    plt.rcParams["axes.linewidth"] = 1
    if darkmode:
        params = ["text.color", "axes.facecolor", "axes.edgecolor", "axes.labelcolor", "xtick.color",
                  "ytick.color", "grid.color", "legend.facecolor", "legend.edgecolor"]
        for p in params:
            plt.rcParams[p] = "white"

    # Which signal to plot
    click.echo("")
    click.echo(click.style("{:#^48}".format(""), fg="green"))
    click.echo(click.style("Plotting",fg="green") + f" {input}")

    ####   READ MAP   #####
    if data: #In case you want to use function directly
        maps_ = [data]
    else:    
        maps_, lmax, outfile, signal_labels = get_map(input, sig, dataset, nside, lmax, fwhm,)
    # Plot all signals specified
    click.echo(click.style("Using signals ",fg="green") + f"{sig}")
    click.echo(click.style("{:#^48}".format(""), fg="green"))
    imgs = [] # Store images for gif
    for i, maps in enumerate(maps_):
        for pl, polt, in enumerate(sig):
            #### Select data column  #####
            m = hp.ma(maps[pl])
            signal_label = signal_labels[polt] if signal_labels else get_signallabel(polt) 
            nsid = hp.get_nside(m)
            #### Smooth  #####
            if float(fwhm) > 0 and input[i].endswith(".fits"):
                click.echo(click.style(f"Smoothing fits map to {fwhm} arcmin fwhm",fg="yellow"))
                m = hp.smoothing(m, fwhm=arcmin2rad(fwhm), lmax=lmax,)
            #### Ud_grade #####
            if nside is not None and input[i].endswith(".fits"):
                if nsid != nside:
                    click.echo(click.style(f"UDgrading map from {nsid} to {nside}", fg="yellow"))
                    m = hp.ud_grade(m, nside,)
            else:
                nside = nsid
            #### Remove monopole or dipole #####
            if remove_dipole or remove_monopole: m = remove_md(m, remove_dipole, remove_monopole, nside)
            #### Scaling factor #####
            if scale:
                if "chisq" in outfile:
                    click.echo(click.style(f"Scaling chisq data with dof={scale}",fg="yellow"))
                    m = (m-scale)/np.sqrt(2*scale)
                else:
                    click.echo(click.style(f"Scaling data by {scale}",fg="yellow"))
                    m *= scale

            #### Automatic variables #####
            if auto:
                (m, ttl, lttl, unt, ticks, cmp, lgscale,) = get_params(m, outfile, signal_label,)

                # Tick bug fix
                mn, md, mx= (ticks[0], None, ticks[-1])
                if not mid and len(ticks)>2:
                    if ticks[0]<ticks[1]  and ticks[-2]<ticks[-1]:
                        md = ticks[1:-1]
                    else:
                        ticks.pop(1)
            else:
                ttl = lttl = unt = ""
                mn = md = mx = None
                ticks = [False, False]
                lgscale = False
                cmp = "planck"

            # Commandline priority
            if logscale != None: lgscale = logscale
            if title: ttl = title
            if ltitle: lttl = ltitle 
            if unit: unt = unit 
            # Get data ticks
            ticks = get_ticks(m, ticks, mn, md, mx, min, mid, max, rng, auto)        
            ticklabels = [fmt(i, 1) for i in ticks]
            #### Logscale ######
            if lgscale: m, ticks = apply_logscale(m, ticks, linthresh=1)
            #### Color map #####
            cmap_ = get_cmap(cmap, cmp,)
            #### Projection ####
            grid_pix, longitude, latitude = project_map(nside, xsize=2000, ysize=int(2000/ 2.0),)
            #### Mask ##########
            grid_map, cmap_ = apply_mask(m, mask, grid_pix, mfill, polt, cmap_) if mask else (m[grid_pix], cmap_)

            click.echo(click.style("Ticks: ", fg="green") + f"{ticklabels}")
            click.echo(click.style("Unit: ", fg="green") + f"{unt}")
            click.echo(click.style("Title: ", fg="green") + f"{ttl}")
            for width in get_sizes(size):
                click.echo(click.style("Size: ", fg="green") + str(width))
                #### Make figure ####
                height = width / 2.0
                if colorbar: height *= 1.275 # Make sure text doesnt change with colorbar
                if gif:
                    # Hacky gif implementation
                    if i == 0:
                        fig = plt.figure(figsize=(cm2inch(width), cm2inch(height),),)
                        ax = fig.add_subplot(111, projection="mollweide")
                else:
                    fig = plt.figure(figsize=(cm2inch(width), cm2inch(height),),)
                    ax = fig.add_subplot(111, projection="mollweide")

                image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=ticks[0], vmax=ticks[-1], rasterized=True, cmap=cmap_, shading='auto',animated=gif)
                # Save img for gif
                if gif: imgs.append([image])
                #### Graticule ####
                if graticule: apply_graticule(ax, width)
                ax.xaxis.set_ticklabels([]); ax.yaxis.set_ticklabels([]) # rm lonlat ticklabs
                #### Colorbar ####
                if colorbar: apply_colorbar(fig, image, ticks, ticklabels, unt, fontsize, linthresh=1, logscale=lgscale)
                #### Right Title ####
                plt.text(4.5, 1.1, r"%s" % ttl, ha="center", va="center", fontsize=labelsize,)
                #### Left Title ####
                plt.text(-4.5, 1.1, r"%s" % lttl, ha="center", va="center", fontsize=labelsize,)
                #### Save ####
                plt.tight_layout()
                if gif: #output gif on last iteration only
                    if i==len(input)-1:
                        output_map(fig, outfile, png, fwhm, colorbar, mask, remove_dipole, darkmode, white_background,cmap_,nside,signal_label,width,outdir, gif, imgs, verbose)
                else:
                    output_map(fig, outfile, png, fwhm, colorbar, mask, remove_dipole, darkmode, white_background,cmap_,nside,signal_label,width,outdir, gif, imgs, verbose)
                    plt.close()
                click.echo("Totaltime:", (time.time() - totaltime),) if verbose else None


def get_params(m, outfile, signal_label,):
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
        ticks = [-3, 0., 3]
        cmap = "RdBu_r"

    # ------ SYNCH ------
    elif tag_lookup(synch_tags, outfile):
        title["comp"] = "s"
        if tag_lookup(synch_beta_tags, outfile+signal_label,):
            click.echo(click.style("{:-^48}".format(f"Detected Synchrotron beta"),fg="yellow"))
            click.echo("Detected Synchrotron beta")
            ticks = [-3.2, -3.15, -3.1]
            title["unit"]  = ""
            title["param"] = r"$\beta$"
            cmap="swamp"
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
            vmin, vmax = get_percentile(m, 97.5)
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
            if "Haslam" in outfile:
                tit = "Haslam"
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
            #ticks = [-100, 0, 100]
            #title["unit"] = r"$\mathrm{MJy/sr}$"
            m = m*2.2703e-6 
            ticks = [-100, 0, 100]
        elif "857" in outfile:
            m *= 2.2703e-6 
            ticks = [-300,0,300]
            #ticks = [-0.05, 0, 0.05]
            #title["unit"] = r"$\mathrm{MJy/sr}$"
        elif "Haslam" in outfile:
            ticks = [-1e4, 0, 1e4]
        

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
        
        vmin, vmax = get_percentile(m, 97.5)
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
        title["custom"] = title["comp"]

        cmap = "bone"
 
    ##
    # Not idenified or ignored #
    ##
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
        vmin, vmax = get_percentile(m, 97.5)
        logscale = False
        #vmin = 0
        ticks = [vmin, vmax]
        #cmap = "planck"
        cmap = "neutral"
    else:
        title["stddev"] = False

    # If signal is an RMS map, add tag.
    if signal_label.endswith("RMS") or "_rms" in outfile:
        click.echo(click.style(f"Detected RMS map",fg="yellow"))
        title["rms"] = True
        vmin, vmax = get_percentile(m, 97.5)
        logscale = False
        #vmin = 0
        ticks = [vmin, vmax]
        #cmap = "planck"
        cmap = "neutral"
    else:
        title["rms"] = False

    if tag_lookup(["diff"], outfile):
        if tag_lookup(["dx12"], outfile):
            title["diff_label"] = "\mathrm{2018}"
        elif tag_lookup(["npipe"], outfile):
            title["diff_label"] = "\mathrm{NPIPE}"
        else:
            title["diff_label"] = ""

        vmin, vmax = get_percentile(m, 97.5)
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

    ttl, lttl = get_title(title)
    unt = title["unit"]
    return (m, ttl, lttl, unt, ticks, cmap, logscale,)

def not_identified(m, signal_label, logscale, title):
    click.echo(click.style("{:-^48}".format(f"Map not recognized, plotting with min and max values"),fg="yellow"))
    title["comp"] = signal_label.split("_")[-1]
    title["param"] = ""
    title["unit"] = ""
    vmin, vmax = get_percentile(m, 97.5)
    ticks = [vmin, vmax]
    cmap = "planck"
    ttl, lttl = get_title(title)
    unt = title["unit"]
    return (m, ttl, lttl, unt, ticks, cmap, logscale,)

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

def get_percentile(m, percentile):
    vmin = np.percentile(m, 100.0 - percentile)
    vmax = np.percentile(m, percentile)

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

def apply_logscale(m, ticks, linthresh):
    click.echo(click.style("Applying semi-logscale", fg="yellow", blink=True, bold=True))
    m = symlog(m,linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i,linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks

def apply_mask(m, mask, grid_pix, mfill, polt, cmap):
    click.echo(click.style(f"Masking using {mask}", fg="yellow"))
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

    return grid_map, cmap

def get_cmap(cmap, cmp):
    # Chose colormap manually
    if cmap == None:
        # If not defined autoset or goto planck
        cmap = cmp
    if "planck" in cmap:
        from pathlib import Path
        if False: #logscale:
            cmap_path = Path(__file__).parent / "planck_cmap_logscale.dat"
        else:
            cmap_path = Path(__file__).parent / "planck_cmap.dat"

        planck_cmap = np.loadtxt(cmap_path) / 255.0            
        if cmap.endswith("_r"):
            planck_cmap = planck_cmap[::-1]

        cmap = col.ListedColormap(planck_cmap, "planck")
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
    return cmap

def project_map(nside, xsize, ysize,):
    theta = np.linspace(np.pi, 0, ysize)
    phi = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))
    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)

    return grid_pix, longitude, latitude

def remove_md(m, remove_dipole, remove_monopole, nside):        
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
    return m

def get_title(_title):
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

    return ttl, lttl

def apply_colorbar(fig, image, ticks, ticklabels, unit, fontsize, linthresh, logscale):
    click.echo(click.style("Applying colorbar", fg="yellow"))
    from matplotlib.ticker import FuncFormatter, LogLocator
    cb = fig.colorbar(image, orientation="horizontal", shrink=0.4, pad=0.04, ticks=ticks, format=FuncFormatter(fmt),)

    cb.ax.set_xticklabels(ticklabels)
    cb.ax.xaxis.set_label_text(unit)
    cb.ax.xaxis.label.set_size(fontsize)
    if logscale:
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
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb

def apply_graticule(ax, width):
    click.echo(click.style("Applying graticule", fg="yellow"))
    #### Formatting ######    
    from matplotlib.projections.geo import GeoAxes
    class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        #shifts labelling by pi
        #Shifts labelling from -180,180 to 0-360
        def __call__(self, x, pos=None):
            if x != 0:
                x *= -1
            if x < 0:
                x += 2 * np.pi
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)

    ax.set_longitude_grid(60)
    ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(60))
    if width < 10:
        ax.set_latitude_grid(45)
        ax.set_longitude_grid_ends(90)            
        ax.grid(True)
            
def get_map(input, sig, dataset, nside, lmax, fwhm,):
    # Get maps array if .h5 file
    signal_labels=None
    maps=[]
    for input in input:
        if input.endswith(".h5"):
            from src.commands_hdf import h5map2fits
            from src.tools import alm2fits_tool
            # Get maps from alm data in .h5
            if dataset.endswith("alm"):
                if not nside:
                    click.echo(click.style("Specify nside for h5 files",fg="red"))
                    sys.exit()

                click.echo(click.style("Converting alms to map",fg="green"))
                (maps_, _, _, _, outfile,) = alm2fits_tool(input, dataset, nside, lmax, fwhm, save=False,)
            # Get maps from map data in .h5
            elif dataset.endswith("map"):
                click.echo(click.style("Reading map from hdf",fg="green"))
                (maps_, _, _, outfile,) = h5map2fits(input, dataset, save=False)

            # Found no data specified kind in .h5
            else:
                click.echo(click.style("Dataset not found. Breaking.",fg="red"))
                click.echo(click.style(f"Does {input}/{dataset} exist?",fg="red"))
                sys.exit()

        elif input.endswith(".fits"):
            maps_, header = hp.read_map(input, field=sig, verbose=False, h=True, dtype=None,)
            header = dict(header)
            signal_labels = []
            for i in range(int(header["TFIELDS"])):
                signal_label = header[f"TTYPE{i+1}"]
                if signal_label in ["TEMPERATURE", "TEMP"]:
                    signal_label = "T"
                if signal_label in ["Q-POLARISATION", "Q_POLARISATION"]:
                    signal_label = "Q"
                if signal_label in ["U-POLARISATION", "U_POLARISATION"]:
                    signal_label = "U"
                signal_labels.append(signal_label)            
            outfile = input.replace(".fits", "")
        else:
            click.echo(click.style("Did not recognize data.",fg="red"))
            sys.exit()

        if maps_.ndim == 1: maps_ = maps_.reshape(1,-1)
        maps.append(maps_)

    return maps, lmax, outfile, signal_labels

def output_map(fig, outfile, png, fwhm, colorbar, mask, remove_dipole, darkmode, white_background,cmap,nside,signal_label,width,outdir,gif,imgs,verbose):
    #### filename ##
    outfile = outfile.replace("_IQU_", "_")
    outfile = outfile.replace("_I_", "_")
    filename = []
    filename.append(f"{str(int(fwhm))}arcmin") if float(fwhm) > 0 else None
    filename.append("cb") if colorbar else None
    filename.append("masked") if mask else None
    filename.append("nodip") if remove_dipole else None
    filename.append("dark") if darkmode else None
    filename.append(f"c-{cmap.name}")

    nside_tag = "_n" + str(int(nside))
    if nside_tag in outfile:
        outfile = outfile.replace(nside_tag, "")

    fn = outfile + f"_{signal_label}_w{str(int(width))}" + nside_tag

    for i in filename:
        fn += f"_{i}"
    filetype = "png" if png else "pdf"
    fn += f".{filetype}"

    starttime = time.time()
    if outdir:
        fn = outdir + "/" + os.path.split(fn)[-1]

    tp = False if white_background else True  
    if gif:
        click.echo(click.style("Outputing GIF:", fg="green") + f" {fn}")
        import matplotlib.animation as animation
        ani = animation.ArtistAnimation(fig, imgs, interval=500, blit=True, repeat_delay=1000)
        ani.save(fn.replace(filetype,"gif"),dpi=300)
    else:
        click.echo(click.style("Outputing PDF:", fg="green") + f" {fn}")
        fig.savefig(fn, bbox_inches="tight", pad_inches=0.02, transparent=tp, format=filetype, dpi=300)
    click.echo("Savefig", (time.time() - starttime),) if verbose else None

def get_ticks(m, ticks, mn, md, mx, min, mid, max, rng, auto):
    # If min and max have been specified, set.
    if rng == "auto" and not auto:
        click.echo(click.style("Setting range from 97.5th percentile of data",fg="yellow"))
        mn, mx = get_percentile(m, 97.5)
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

    return [float(i) for i in ticks]