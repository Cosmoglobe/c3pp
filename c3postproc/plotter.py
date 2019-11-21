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

def Plotter(flags=None):
    rcParams['backend'] = 'pdf' #
    rcParams['legend.fancybox'] = True
    rcParams['lines.linewidth'] = 2
    rcParams['savefig.dpi'] = 300
    rcParams['axes.linewidth'] = 1
    # use of Sans Serif also in math mode
    darkmode = False
    masked   = False

    if "-darkmode" in flags:
        darkmode = True
        rcParams['text.color']   = 'white'   # axes background color
        rcParams['axes.facecolor']   = 'white'   # axes background color
        rcParams['axes.edgecolor' ]  = 'white'   # axes edge color
        rcParams['axes.labelcolor']   = 'white'
        rcParams['xtick.color']    = 'white'      # color of the tick labels
        rcParams['ytick.color']    = 'white'      # color of the tick labels
        rcParams['grid.color'] =   'white'   # grid color
        rcParams['legend.facecolor']    = 'inherit'   # legend background color (when 'inherit' uses axes.facecolor)
        rcParams['legend.edgecolor']= 'white' # legend edge color (when 'inherit' uses axes.edgecolor)

    rc('text.latex', preamble=r'\usepackage{sfmath}')

     # Which signal to plot
    map_= flags[0] # Map is always first flag
    pollist = get_pollist(flags)
    signal_labels = ["I", "Q", "U"]

    print()
    print("Plotting " + map_)

    #######################
    ####   READ MAP   #####
    #######################
    
    if ".fits" in map_[-5:]:
        # Make sure its the right shape for indexing
        # This is a really dumb way of doing it
        idx = tuple(pollist)
        dats = [0, 0, 0]
        for i in idx:
            dats[i] = hp.ma(hp.read_map(map_, field=i))
            nside = hp.npix2nside(len(dats[i]))
        maps = np.array(dats)
        outfile = map_.replace(".fits", "")

    elif ".h5" in map_[-3:]:
        import h5py
        dataset =  get_key(flags, map_) 
        with h5py.File(map_, 'r') as f:
            maps = f[dataset][()]
            lmax = f[dataset[:-4]+"_lmax"][()] # Get lmax from h5

        if "alm" in dataset[-3:]:
            alms = np.array(maps,dtype=np.complex128)
            # Convert alms to map
            print("Converting alms to map")
            
            
            if "-lmax" in flags:
                lmax = int(get_key(flags, "-lmax"))
                print("Setting lmax to ", lmax)
                mmax = lmax
            else:
                # Let alm2map chose
                # This does NOT work
                #lmax = None
                mmax = lmax

            if "-fwhm" in flags:
                fwhm = float(get_key(flags, "-fwhm"))
            else:
                fwhm = 0.0

            if "-lmin" in flags:
                lmin = int(get_key(flags, "-lmin"))
            else:
                lmin = 0

            nside = int(get_key(flags, dataset))
            
            # does the +lmin make sense here?
            hehe = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1) + lmin

            maps = hp.sphtfunc.alm2map(alms[:,lmin:hehe], nside, lmax=lmax, fwhm=arcmin2rad(fwhm))
            outfile =  dataset.replace("/", "_")
            outfile = outfile.replace("_alm","")
        elif "map" in dataset[-3:]:
            print("Reading map from h5")
            nside = hp.npix2nside(maps.shape[-1])
            outfile =  dataset.replace("/", "_")
            outfile = outfile.replace("_map","")

    print("nside", nside, "total file shape", maps.shape)

    for polt in pollist:
        m = maps[polt] # Select map signal (I,Q,U)
        #m= m[hp.ring2nest(nside, range(12*(nside)**2))]
        
        #######################
        #### Auto-param   #####
        #######################
        title, ticks, ticklabels, unit, coltype, color, logscale = get_params(m, outfile, polt, signal_labels)
        vmin = ticks[0]
        vmax = ticks[-1]
        tmin = ticklabels[0]
        tmax = ticklabels[-1]
        
        min, max = get_range(flags)
        if min != None:
            vmin = min
            tmin = str(min)
        if max != None:
            vmax = max
            tmin = str(max)

        ##########################
        #### Plotting Params #####
        ##########################
        unit       = unit
        title      = title        # Short name upper right
        colorbar   = 1 if "-bar" in flags else 0           # Colorbar on

        # Image size -  ratio is always 1/2
        xsize = 2000
        ysize = int(xsize/2.)

        #######################
        ####   logscale   #####
        #######################
        # Some maps turns on logscale automatically
        if "-logscale" in flags or logscale: 
            m = np.log10(0.5*(m+np.sqrt(4.+m*m)))
            m = np.maximum(np.minimum(m,vmax),vmin)

        ######################
        #### COLOR SETUP #####
        ######################

        from matplotlib.colors import ListedColormap

        if coltype == 0:
            startcolor = 'black'
            midcolor = color
            endcolor = 'white'
            if color == "none" :
                cmap = col.LinearSegmentedColormap.from_list('own2',[startcolor,endcolor])
            elif color == "special":
                cmap = col.LinearSegmentedColormap.from_list('own2',[endcolor,col1,col2, startcolor, col3,col4, endcolor])
            elif color == "special2":
                cmap = col.LinearSegmentedColormap.from_list('own2',[endcolor,col1, startcolor, col2, endcolor])
            else:
                cmap = col.LinearSegmentedColormap.from_list('own2',[startcolor,midcolor,endcolor])
        elif coltype == 1:
            cmap = plt.get_cmap(color)
        elif coltype == 2:
            #cmap = get_cmbcolormap()
            cmap = ListedColormap(np.loadtxt(color)/255.)

        #cmap.set_bad("gray") # color of missing pixels
        #cmap.set_under("white") # color of background, necessary if you want to use
        # using directly matplotlib instead of mollview has higher quality output

        ######################
        ####  Projection #####
        ######################
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
        if "-mask" in flags:
            masked = True
            mask_name = get_key(flags, "-mask")
            m.mask = np.logical_not(hp.read_map(mask_name,1))
            grid_mask = m.mask[grid_pix]
            grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)
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
                    x += 2*np.pi
                return GeoAxes.ThetaFormatter.__call__(self, x, pos)



        sizes = get_sizes(flags)
        for width in sizes:
            print("Plotting size " + str(width))
            height = width/2.
            height *=  1.275 if colorbar else 1

            ################
            ##### font #####
            ################
            if width > 12.:
                fontsize=8
            elif width == 12.:
                fontsize=7 

            else:
                fontsize=6



            fig = plt.figure(figsize=(cm2inch(width), cm2inch(height)))

            ax = fig.add_subplot(111,projection='mollweide')


            # rasterized makes the map bitmap while the labels remain vectorial
            # flip longitude to the astro convention
            image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)

            # graticule
            ax.set_longitude_grid(60)
            ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(60))

            if width < 10:
                ax.set_latitude_grid(45)
                ax.set_longitude_grid_ends(90)
            
        
            
            ################
            ### COLORBAR ###
            ################
            if colorbar == 1:
                # colorbar
                cb = fig.colorbar(image, orientation='horizontal', shrink=.3, pad=0.08, ticks=ticks, format=ticker.FuncFormatter(fmt))
                if tmax != False or tmin != False: # let function format if not autoset
                    cb.ax.set_xticklabels(ticklabels)
                cb.ax.xaxis.set_label_text(unit) 
                cb.ax.xaxis.label.set_size(fontsize)
                cb.ax.minorticks_on()
                cb.ax.tick_params(which='both', axis='x', direction='in', labelsize=fontsize)
                cb.ax.xaxis.labelpad = 4 #-11
                # workaround for issue with viewers, see colorbar docstring
                cb.solids.set_edgecolor("face")

                #ax.tick_params(axis='x', labelsize=10)
            #ax.tick_params(axis='y', labelsize=10)

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
            plt.text(6.,  1.3, r"%s" % title, ha='center', va='center', fontsize=fontsize)

            ##############
            #### SAVE ####
            ##############
            plt.tight_layout()
            filetype = ".png" if "-png" in flags else ".pdf"
            tp = False if "-white_background" in flags else True # Turn on transparency unless told otherwise

            ##############
            ## filename ##
            ##############
            filename = []
            filename.append('{}arcmin'.format(str(int(fwhm)))) if "-fwhm" in flags else None
            filename.append('cb') if "-bar" in flags else None
            filename.append('masked') if "-mask" in flags else None
            filename.append('dark') if "-darkmode" in flags else None
        
            fn = outfile+"_"+signal_labels[polt]+"_w"+str(int(width))+"_n"+str(int(nside))
            for i in filename:
                fn += "_"+i
            fn += filetype

            plt.savefig(fn, bbox_inches='tight',  pad_inches=0.02, transparent=tp)
            plt.close()


def get_params(m, outfile, polt, signal_labels):
    print()
    logscale = False
    
    # Everything listed here will be recognized
    # If tag is found in output file, use template
    cmb_tags = ["cmb"]
    chisq_tags = ["chisq"]
    synch_tags = ["synch_c", "synch_amp"]
    dust_tags  = ["dust_c", "dust_amp"]
    ame_tags   = ["ame_c", "ame_amp", "ame1_c", "ame1_amp"]
    ff_tags    = ["ff_c", "ff_amp"]
    co10_tags    = ["co10", "co-100"]
    co21_tags    = ["co21", "co-217"]
    co32_tags    = ["co32", "co-353"]
    hcn_tags    = ["hcn"]
    dust_T_tags = ["dust_T", "dust_Td"]
    dust_beta_tags = ["dust_beta"]
    synch_beta_tags = ["synch_beta"]
    ff_Te_tags = ["ff_T_e", "ff_Te"]
    ff_EM_tags = ["ff_EM"]
    res_tags = ["residual_", "res_"]

    if tag_lookup(cmb_tags, outfile):
        print("Plotting CMB " + signal_labels[polt] )

        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{CMB}}$"

        if polt>0:
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
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{CMB}}$"

        coltype = 2
        from pathlib import Path
        color = Path(__file__).parent / 'parchment1.dat'

    elif tag_lookup(chisq_tags, outfile):
        
        title = r"$\chi^2$ " + signal_labels[polt]

        if polt>0:
            vmin = 0
            vmax = 32
        else:
            vmin = 0
            vmax = 76

        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]
        
        print("Plotting chisq with vmax = " + str(vmax) +" "+ signal_labels[polt])

        unit = ""
        coltype = 0
        color = "none"

    elif tag_lookup(synch_tags, outfile):
        print("Plotting Synchrotron" +" "+ signal_labels[polt])
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{s}}$ "
        print("Applying logscale (Rewrite if not)")
        if polt>0:
            vmin = -1.69
            vmax = 1.69
            tmin = str(-50)        
            tmax = str(50)
            logscale = True

            vmid = 0
            tmid = "0"
            ticks = [vmin, vmid, vmax]
            ticklabels = [tmin,tmid,tmax]

            coltype = 0
            color = "special2"
            col1="darkgoldenrod"
            col2="darkgreen"
        else:
            vmin = 1
            vmax = np.log10(100)
            tmin = str(10)
            tmax = str(100)
            logscale=True
            coltype =   0

            ticks = [vmin, vmax]
            ticklabels = [tmin, tmax]

        color = "green"

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"

    elif tag_lookup(ff_tags, outfile):
        print("Plotting freefree")
        print("Applying logscale (Rewrite if not)")

        vmin = 0 #0
        vmid = np.log10(100)
        vmax = np.log10(10000) #1000
        
        tmin = str(0)
        tmid = str(r'$10^2$')
        tmax = str(r'$10^4$')

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{ff}}$"
        logscale=True
        coltype = 0
        color = "Navy"

    elif tag_lookup(dust_tags, outfile):
        print("Plotting Thermal dust" +" "+ signal_labels[polt])
        print("Applying logscale (Rewrite if not)")
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{d}}$ " 
        if polt>0:
            vmin = -2
            vmid = 0
            vmax = 2

            tmin = str(-100)
            tmid = 0
            tmax = str(100)

            logscale=True
            coltype= 0
            color = "special"
            col1="deepskyblue"
            col2="blue"
            col3="firebrick"
            col4="darkorange"
        else:
            vmin = 0
            vmid = 2
            vmax = 4 #1000

            tmin = str(0)
            tmid = str(r"$10^2$")
            tmax = str(r"$10^4$")

            logscale=True
            coltype= 1
            color = "gist_heat"


        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"

    elif tag_lookup(ame_tags, outfile):
        print("Plotting AME")
        print("Applying logscale (Rewrite if not)")
       
        vmin = 0 #0
        vmid = np.log10(100)
        vmax = np.log10(10000) #1000
        
        tmin = str(0)
        tmid = str(r'$10^2$')
        tmax = str(r'$10^4$')

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mu\mathrm{K}_{\mathrm{RJ}}$"
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{ame}}$"
        logscale=True
        coltype=0
        color="DarkOrange"

    elif tag_lookup(co10_tags, outfile):
        print("Plotting CO10")
        print("Applying logscale (Rewrite if not)")
        vmin = 0
        vmid = 1
        vmax = 2

        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{CO10}}$"
        logscale=True
        coltype=1
        color="gray"


    elif tag_lookup(co21_tags, outfile):
        print("Plotting CO21")
        print("Applying logscale (Rewrite if not)")
        vmin = 0
        vmid = 1
        vmax = 2
        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{CO21}}$"
        logscale=True
        coltype=1
        color="gray"

    elif tag_lookup(co32_tags, outfile):
        print("Plotting 32")
        print("Applying logscale (Rewrite if not)")
        vmin = 0
        vmid = 1
        vmax = 2 #0.5
        tmin = str(0)
        tmid = str(10)
        tmax = str(100)

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{CO32}}$"
        logscale=True
        coltype=1
        color="gray"

    elif tag_lookup(hcn_tags, outfile):
        print("Plotting HCN")
        print("Applying logscale (Rewrite if not)")
        vmin = -14
        vmax = -10
        tmin = str(0.01)
        tmax = str(100)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = r"$"+ signal_labels[polt]+"$" + r"$_{\mathrm{HCN}}$"
        logscale=True
        coltype=1
        color="gray"

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
        coltype=1
        color="bone"

    # SPECTRAL PARAMETER MAPS
    elif tag_lookup(dust_T_tags, outfile):
        print("Plotting Thermal dust Td")

        title =  r"$"+ signal_labels[polt]+"$" + r"$T_d$ "

        vmin = 14
        vmax = 30
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        coltype=1
        color="bone"

    elif tag_lookup(dust_beta_tags, outfile):
        print("Plotting Thermal dust beta")

        title =  r"$"+ signal_labels[polt]+"$" + r"$\beta_d$ "

        vmin = 1.45
        vmax = 1.55
        tmin = str(vmin)
        tmax = str(vmax)
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        coltype=1
        color="bone"

    elif tag_lookup(synch_beta_tags, outfile):
        print("Plotting Synchrotron beta")

        title = r"$"+ signal_labels[polt]+"$" +  r"$\beta_s$ " 

        vmin = -3.15
        vmax = -3.12
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        coltype=1
        color="bone"

    elif tag_lookup(ff_Te_tags, outfile):
        print("Plotting freefree T_e")

        vmin = 5000
        vmax = 8000
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"
        coltype=1
        color="bone"

    elif tag_lookup(ff_EM_tags, outfile):
        print("Plotting freefree EM MIN AND MAX VALUES UPDATE!")

        vmin = 5000
        vmax = 8000
        tmin = str(vmin)
        tmax = str(vmax)

        vmax = np.percentile(m,95)
        vmin = np.percentile(m,5)
        tmin = False
        tmax = False
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        title = r"$T_{e}$"
        coltype=1
        color="bone"


    #################
    # RESIDUAL MAPS #
    #################

    elif tag_lookup(res_tags, outfile):
        print("Plotting residual map" +" "+ signal_labels[polt])

        if "res_" in outfile:
            tit  = str(re.findall(r'res_smooth3idual_(.*?)_',outfile)[0])
        else:
            tit = str(re.findall(r'residual_(.*?)_',outfile)[0])

        title =  r"$"+ signal_labels[polt]+"$" +  r"{} GHz ".format(tit)

        vmin = -10
        vmid = 0
        vmax = 10
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)

        unit =  r"$\mu\mathrm{K}$"
        coltype=2

        from pathlib import Path
        color = Path(__file__).parent / 'parchment1.dat'

        if "545" in outfile:
            vmin = -1e2
            vmax = 1e2
            tmin = str(vmin)
            tmax = str(vmax)
            unit = r"$\mathrm{MJy/sr}$"
        elif "857" in outfile:
            vmin = -0.05 #-1e4
            vmax = 0.05 #1e4
            tmin = str(vmin)
            tmax = str(vmax)
            unit = r"$\mathrm{MJy/sr}$"

        ticks = [vmin, vmid, vmax]
        ticklabels = [tmin,tmid,tmax]

    else:
        print("Map not recognized, plotting with min and max values")
        vmax = np.percentile(m,95)
        vmin = np.percentile(m,5)
        tmin = False
        tmax = False
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]
        unit = ""
        title =  r"$"+ signal_labels[polt]+"$"
        coltype=2
        from pathlib import Path
        color = Path(__file__).parent / 'parchment1.dat'

    return title, ticks, ticklabels, unit, coltype, color, logscale

def get_pollist(flags):
    pollist = []
    if "-I" in flags:
        pollist.append(0)
    if "-Q" in flags:
        pollist.append(1)
    if "-U" in flags:
        pollist.append(2)
    if "-QU" in flags:
        pollist.extend((1, 2))
    if "-IU" in flags:
        pollist.extend((0, 2))
    if "-IQU" in flags:
        pollist.extend((0, 1, 2))
    if len(pollist) == 0:
        pollist = [0]
    return pollist

def get_sizes(flags):
    sizes = []
    if "-small" in flags:
        sizes.append(8.8)
    if "-medium" in flags:
        sizes.append(12.0)
    if "-large" in flags:
        sizes.append(18.0)
    if len(sizes) == 0:
        sizes = [8.8, 12.0, 18.0]
    return sizes

def get_range(flags):
    min = float(get_key(flags, "-min")) if "-min" in flags else None
    max = float(get_key(flags, "-max")) if "-max" in flags else None
    if "-range" in flags:
        r = float(get_key(flags, "-range"))
        min = -r
        max =  r
    return min, max

def get_key(flags, keyword):
    return flags[flags.index(keyword) + 1]

def fmt(x, pos):
    '''
    Format color bar labels
    '''
    if abs(x) > 1e4:
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \cdot 10^{{{}}}$'.format(a, b)
    elif abs(x) > 1e2:
        return r'${:d}$'.format(int(x))
    elif abs(x) > 1e1:
        return r'${:.1f}$'.format(x)
    else: 
        return r'${:.2f}$'.format(x)

def cm2inch(cm):
    return cm*0.393701

def tag_lookup(tags, outfile):
    return any(e in outfile for e in tags)

def arcmin2rad(arcmin):
    return arcmin*(2*np.pi)/21600