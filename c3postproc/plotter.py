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
    unit,
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
                    signal_label = header[f'TTYPE{polt+1}']
                except:
                    pass

                m = hp.ma(map) # Dont use header for this
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
        tempunit = unit
        templogscale = logscale
        tempcmap = cmap

    
        # ttl, unt and cmb are temporary variables for title, unit and colormap
        if auto:
            ttl, ticks, ticklabels, unt, cmp, lgscale, format_ticks= get_params(
                m, outfile, polt,
            )
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
                format_ticks = False
                if minmax:
                    mn = np.min(m)
                    mx = np.max(m)
                else:
                    mx = np.percentile(m, 97.5)
                    mn = np.percentile(m, 2.5)
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
            ticklabels[0] = str(min)

        if max is not False:
            ticks[-1] = float(max)
            ticklabels[-1] = str(max)

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
        elif cmap == "turbo":
            turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
            cmap=col.ListedColormap(turbo_colormap_data, "turbo")
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

                # Don't format ticks if autoset
                if format_ticks:
                   cb.ax.set_xticklabels(ticklabels)

                cb.ax.xaxis.set_label_text(unit)
                cb.ax.xaxis.label.set_size(fontsize)
                #cb.ax.minorticks_on()

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

            #############
            ## TITLE ####
            #############
            plt.text(
                6.0, 1.3, r"%s" % title, ha="center", va="center", fontsize=fontsize,
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
            
            outfile = outfile.replace("_IQU_","_")
            outfile = outfile.replace("_I_","_")   

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

def get_params(m, outfile, polt):
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
    dust_T_tags = ["dust_T", "dust_Td"]
    dust_beta_tags = ["dust_beta"]
    synch_beta_tags = ["synch_beta"]
    ff_Te_tags = ["ff_T_e", "ff_Te"]
    ff_EM_tags = ["ff_EM"]
    res_tags = ["residual_", "res_"]
    tod_tags = ["Smap"]
    freqmap_tags = ["BP_030", "BP_044", "BP_070"]
    ignore_tags = ["radio_"]

    sl = get_signallabel(polt)
    startcolor = 'black'
    endcolor = 'white'
    format_ticks = True # If min and max are autoset, dont do this.

    if tag_lookup(cmb_tags, outfile,):

        print(f"Plotting CMB signal {sl}")
        
        title = r"$" + sl + "$" + r"$_{\mathrm{CMB}}$"

        if polt%3 > 0:
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

        cmap = "planck"

    elif tag_lookup(chisq_tags, outfile):
        title = r"$\chi^2$ " + sl

        if polt%3 > 0:
            vmin = 0
            vmax = 32
        else:
            vmin = 0
            vmax = 76

        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        print("Plotting chisq with vmax = " + str(vmax) + " " + sl)

        unit = ""
        cmap = col.LinearSegmentedColormap.from_list("BkWh", ["black", "white"])

    elif tag_lookup(synch_tags, outfile):
        print(f"Plotting Synchrotron {sl}")
        title = r"$" + sl + "$" + r"$_{\mathrm{s}}$ "
        if polt%3 > 0:
            # BP uses 30 GHz ref freq for pol
            vmin = -np.log10(50)
            vmax = np.log10(50)
            tmin = str(-50)
            tmax = str(50)
            logscale = True

            vmid = 0
            tmid = "0"
            ticks = [vmin, vmid, vmax]
            ticklabels = [tmin, tmid, tmax]

            
            #col1 = "darkgoldenrod"
            #col2 = "darkgreen"
            #cmap = col.LinearSegmentedColormap.from_list("YlBkGr",
            #    [endcolor, col1, startcolor, col2, endcolor], 
            #)
            cmap = "planck"

            unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        else:
            # BP uses 408 MHz GHz ref freq
            vmin = np.log10(10*10**6)
            vmid1 = np.log10(30*10**6)
            vmid2 = np.log10(100*10**6)
            vmax = np.log10(300*10**6)

            tmin = str(r"$10$")
            tmid1 = str(r"$30$")
            tmid2 = str(r"$100$")
            tmax = str(r"$300$")
    
            logscale = True
            #cmap = col.LinearSegmentedColormap.from_list("BkGrWh",
            #    ["black", "green", "white"], 
            #)
            cmap = "planck"


            ticks = [vmin,vmid1, vmid2, vmax,]
            ticklabels = [tmin, tmid1, tmid2, tmax, ]

            unit = r"$\mathrm{K}_{\mathrm{RJ}}$"

    elif tag_lookup(ff_tags, outfile):
        print("Plotting freefree")

        vmin = 0  # 0
        vmid = np.log10(100)
        vmax = np.log10(10000)  # 1000

        tmin = str(0)
        tmid = str(r"$10^2$")
        tmax = str(r"$10^4$")

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{ff}}$"
        logscale = True

        #  cmap = col.LinearSegmentedColormap.from_list("BkBlWh",["black", "Navy", "white"], )
        cmap = "planck"

    elif tag_lookup(dust_tags, outfile):
        print("Plotting Thermal dust" + " " + sl)
        title = r"$" + sl + "$" + r"$_{\mathrm{d}}$ "
        if polt%3 > 0:
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
            cmap = col.LinearSegmentedColormap.from_list("BlBkRd",
                [endcolor, col1, col2, startcolor, col3, col4, endcolor], 
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
        print("Plotting AME")

        vmin = 0  # 0
        vmid = np.log10(100)
        vmax = np.log10(10000)  # 1000

        tmin = r"$0$"
        tmid = r"$10^2$"
        tmax = r"$10^4$"

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{ame}}$"
        logscale = True
        #cmap = col.LinearSegmentedColormap.from_list("BkOrWh",
        #    ["black", "DarkOrange", "white"], 
        #)
        cmap = "planck"


    elif tag_lookup(co10_tags, outfile):
        print("Plotting CO10")
        vmin = 0
        vmid = np.log10(10)
        vmax = np.log10(100)

        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CO10}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(co21_tags, outfile):
        print("Plotting CO21")
        vmin = 0
        vmid = 1
        vmax = 2
        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CO21}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(co32_tags, outfile):
        print("Plotting 32")
        vmin = 0
        vmid = 1
        vmax = 2  # 0.5
        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{CO32}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(hcn_tags, outfile):
        print("Plotting HCN")
        vmin = -14
        vmax = -10
        tmin = str(0.01)
        tmax = str(100)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$" + sl + "$" + r"$_{\mathrm{HCN}}$"
        logscale = True
        cmap = plt.get_cmap("gray")

    elif tag_lookup(ame_tags, outfile):
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

    # SPECTRAL INDEX MAPS
    elif tag_lookup(dust_T_tags, outfile):
        print("Plotting Thermal dust Td")

        title = r"$" + sl + "$ " + r"$T_d$ "

        vmin = 14
        vmax = 30
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        cmap = plt.get_cmap("bone")

    elif tag_lookup(dust_beta_tags, outfile):
        print("Plotting Thermal dust beta")

        title = r"$" + sl + "$ " + r"$\beta_d$ "

        vmin = 1.3
        vmax = 1.8
        tmin = str(vmin)
        tmax = str(vmax)
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        cmap = plt.get_cmap("bone")

    elif tag_lookup(synch_beta_tags, outfile):
        print("Plotting Synchrotron beta")

        title = r"$" + sl + "$ " + r"$\beta_s$ "

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
        print("Plotting freefree EM MIN AND MAX VALUES UPDATE!")

        vmax = np.percentile(m, 97.5)
        vmin = np.percentile(m, 2.5)

        tmid = str(vmid)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        format_ticks = False

        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"
        cmap = plt.get_cmap("bone")

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

        vmin = -10
        vmid = 0
        vmax = 10
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)

        unit = r"$\mu\mathrm{K}$"
        cmap = "planck"

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

    ############
    # TOD MAPS #
    ############

    elif tag_lookup(tod_tags, outfile):
        from re import findall

        print("Plotting Smap map" + " " + sl)

        tit = str(findall(r"tod_(.*?)_Smap", outfile)[0])
        title = fr"{tit} " + r"  $" + sl + "$"

        vmin = -0.2
        vmid = 0
        vmax = 0.2
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)

        unit = r"$\mu\mathrm{K}$"
        cmap = "planck"

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]
    ############
    # FREQMAPS #
    ############

    elif tag_lookup(freqmap_tags, outfile):
        from re import findall

        print("Plotting Frequency map" + " " + sl)

        tit = str(findall(r"BP_(.*?)_", outfile)[0])
        title = fr"{tit} " + r"  $" + sl + "$"

        vmax = np.percentile(m, 97.5)
        vmid = 0.0
        vmin = np.percentile(m, 2.5)
    
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)
        format_ticks = False

        unit = r"$\mu\mathrm{K}$"
        
        cmap ="planck"

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin, tmid, tmax]

    ############################
    # Not idenified or ignored #
    ############################
    elif tag_lookup(ignore_tags, outfile):
        print(
            f'{outfile} is on the ignore list, under tags {ignore_tags}. Remove from "ignore_tags" in plotter.py. Breaking.'
        )
        sys.exit()
    else:
        print("Map not recognized, plotting with min and max values")
        vmax = np.percentile(m, 97.5)
        vmin = np.percentile(m, 2.5)
    
        tmin = str(vmin)
        tmax = str(vmax)
        format_ticks = False

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]
        unit = ""
        title = r"$" + sl + "$"

        cmap = "planck"

    return title, ticks, ticklabels, unit, cmap, logscale, format_ticks

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


def fmt(x, pos):
    """
    Format color bar labels
    """
    if abs(x) > 1e4:
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        return fr"${a} \cdot 10^{{{b}}}$"
    elif abs(x) > 1e2:
        return fr"${int(x):d}$"
    elif abs(x) > 1e1:
        return fr"${x:.1f}$"
    elif abs(x) == 0.0:
        return fr"${x:.1f}$"
    else:
        return fr"${x:.2f}$"


def cm2inch(cm):
    return cm * 0.393701


def tag_lookup(tags, outfile):
    return any(e in outfile for e in tags)
