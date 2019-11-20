#!/Users/svalheim/anaconda2/bin/python
#-*- mode: python -*-

import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import healpy as hp
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as col
import re
from matplotlib import rcParams, rc
import matplotlib.ticker as ticker

def cm2inch(cm):
    return cm*0.393701

def Plotter(map_, optn_flags=None, png=False):
    rcParams['backend'] = 'pdf' #
    rcParams['legend.fancybox'] = True
    rcParams['lines.linewidth'] = 2
    rcParams['savefig.dpi'] = 300
    rcParams['axes.linewidth'] = 1
    rcParams['axes.titlesize'] = 'x-large'
    rcParams['axes.labelsize'] = 10 #'large'
    rcParams['legend.fontsize'] = 10 #
    rcParams['xtick.labelsize'] = 10 #
    rcParams['ytick.labelsize'] = 10 #
    rcParams['xtick.major.pad'] = 6 #
    rcParams['ytick.major.pad'] = 6 #
    rcParams['font.size'] = 10 #
    # use of Sans Serif also in math mode
    darkmode = False
    masked   = False

    if "-darkmode" in optn_flags:
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

    print(map_, optn_flags )
     # Which signal to plot
    pollist = get_pollist(optn_flags)
    signal_labels = ["I", "Q", "U"]
    for polt in pollist:
        print()
        print("Plotting " + str(map_))

        #######################
        ####   READ MAP   #####
        #######################
        map_=str(map_)
        scale = 1 # Scale by a factor of
        filename = str(map_)

        m = hp.ma(hp.read_map(filename,field=(polt)))*scale
        nside = hp.npix2nside(len(m))

        #######################
        #### Auto-param   #####
        #######################
        title, ticks, ticklabels, unit, coltype, color, logscale = get_params(m, map_, polt, signal_labels)
        vmin = ticks[0]
        vmax = ticks[-1]
        tmin = ticklabels[0]
        tmax = ticklabels[-1]
        
        min, max = get_range(optn_flags)
        if min != None:
            vmin = min
            tmin = str(min)
        if max != None:
            vmax = max
            tmin = str(max)

        ##########################
        #### Plotting Params #####
        ##########################
        outfile = map_.replace(".fits", "")
        unit       = unit
        title      = title        # Short name upper right
        colorbar   = 1 if "-bar" in optn_flags else 0           # Colorbar on

        # Image size -  ratio is always 1/2
        xsize = 2000
        ysize = int(xsize/2.)

        #######################
        ####   logscale   #####
        #######################
        # Some maps turns on logscale automatically
        if "-logscale" in optn_flags or logscale: 
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
        if "-mask" in optn_flags:
            masked = True
            mask_name = get_key(optn_flags, "-mask")
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



        sizes = get_sizes(optn_flags)
        for width in sizes:
            print("Plotting size " + str(width))

            height = width/2.
            if colorbar: # Same text size. (Hardcoded for 12)
                height = height + 1.65

            fig = plt.figure(figsize=(cm2inch(width), cm2inch(height)))

            ax = fig.add_subplot(111,projection='mollweide')
            # remove white space around the image
            #plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

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
                cb.ax.minorticks_on()
                cb.ax.tick_params(which='both', axis='x', direction='in')
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
            if width > 12.:
                plt.text(6.,  1.3, r"%s" % title, ha='center', va='center', fontsize=8)
            elif width == 12.:
                plt.text(6.,  1.3, r"%s" % title, ha='center', va='center', fontsize=7)
            else:
                plt.text(6.,  1.3, r"%s" % title, ha='center', va='center', fontsize=6)

            ##############
            #### SAVE ####
            ##############
            plt.tight_layout()
            filetype = ".png" if png else ".pdf"
            tp = False if "-white_background" in optn_flags else True # Turn on transparency unless told otherwise

            ##############
            ## filename ##
            ##############
            filename = []
            filename.append('masked') if "-mask" in optn_flags else None
            filename.append('darkmode') if "-darkmode" in optn_flags else None
            fn = outfile+"_"+signal_labels[polt]+"_w"+str(int(width))
            for i in filename:
                fn += "_"+i
            fn += filetype

            plt.savefig(fn, bbox_inches='tight',  pad_inches=0.02, transparent=tp)
            plt.close()


def get_params(m, map_, polt, signal_labels):
    print()
    logscale = False
    # AMPLITUDE MAPS
    if "cmb" in map_:
        print("Plotting CMB " + signal_labels[polt])

        title = "$A_{\mathrm{CMB}}$ " + signal_labels[polt]

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
        path = os.path.dirname(os.path.abspath(__file__))
        color = path+"/parchment1.dat"

    elif "chisq" in map_:
        
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

    elif "synch_c" in map_:
        print("Plotting Synchrotron" +" "+ signal_labels[polt])
        title = r"$A_{\mathrm{s}}$ "  + signal_labels[polt]
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

    elif "ff_" in map_:
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
        title = r"$A_{\mathrm{ff}}$"
        logscale=True
        coltype = 0
        color = "Navy"

    elif "dust_c" in map_:
        print("Plotting Thermal dust" +" "+ signal_labels[polt])
        print("Applying logscale (Rewrite if not)")
        title = r"$A_{\mathrm{d}}$ "  + signal_labels[polt]

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

    elif "ame1_c" in map_ or "ame_" in map_:
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
        title = r"$A_{\mathrm{ame}}$"
        logscale=True
        coltype=0
        color="DarkOrange"

    elif "co-100" in map_ or "co10" in map_:
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
        title = "$A_{\mathrm{CO10}}$"
        logscale=True
        coltype=1
        color="gray"


    elif "co-217" in map_ or "co21" in map_:
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
        title = "$A_{\mathrm{CO21}}$"
        logscale=True
        coltype=1
        color="gray"

    elif "co-353" in map_ or "co32" in map_:
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
        title = "$A_{\mathrm{CO32}}$"
        logscale=True
        coltype=1
        color="gray"

    elif "hcn_c" in map_:
        print("Plotting HCN")
        print("Applying logscale (Rewrite if not)")
        vmin = -14
        vmax = -10
        tmin = str(0.01)
        tmax = str(100)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}_{\mathrm{RJ}}\, \mathrm{km}/\mathrm{s}$"
        title = "$A_{\mathrm{HCN}}$"
        logscale=True
        coltype=1
        color="gray"

        # SPECTRAL PARAMETER MAPS
    elif "ame1_nu" in map_:
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

    elif "dust_Td" in map_:
        print("Plotting Thermal dust Td")

        title = r"$T_d$ "  + signal_labels[polt]

        vmin = 14
        vmax = 30
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = r"$\mathrm{K}$"
        coltype=1
        color="bone"

    elif "dust_beta" in map_:
        print("Plotting Thermal dust beta")

        title = r"$\beta_d$ " + signal_labels[polt]

        vmin = 1.45
        vmax = 1.55
        tmin = str(vmin)
        tmax = str(vmax)
        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        coltype=1
        color="bone"

    elif "synch_beta" in map_:
        print("Plotting Synchrotron beta")

        title = r"$\beta_s$ " + signal_labels[polt]

        vmin = -3.15
        vmax = -3.12
        tmin = str(vmin)
        tmax = str(vmax)

        ticks = [vmin, vmax]
        ticklabels = [tmin, tmax]

        unit = ""
        coltype=1
        color="bone"

    elif "ff_T_e" in map_:
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

    #################
    # RESIDUAL MAPS #
    #################

    elif "residual_" in map_ or "res_" in map_:
        print("Plotting residual map" +" "+ signal_labels[polt])

        if "res_" in map_:
            tit  = str(re.findall(r'res_smooth3idual_(.*?)_',map_)[0])
        else:
            tit = str(re.findall(r'residual_(.*?)_',map_)[0])

        title = r"{} GHz ".format(tit) + signal_labels[polt]

        vmin = -10
        vmid = 0
        vmax = 10
        tmin = str(vmin)
        tmid = str(vmid)
        tmax = str(vmax)


        unit =  r"$\mu\mathrm{K}$"
        coltype=2

        path = os.path.dirname(os.path.abspath(__file__))
        color = path+"/parchment1.dat"

        if "545" in map_:
            vmin = -1e2
            vmax = 1e2
            tmin = str(vmin)
            tmax = str(vmax)
            unit = r"$\mathrm{MJy/sr}$"
        elif "857" in map_:
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
        title = ""
        coltype=2
        path = os.path.dirname(os.path.abspath(__file__))
        color = path+"/parchment1.dat"

    return title, ticks, ticklabels, unit, coltype, color, logscale

def get_pollist(optn_flags):
    pollist = []
    if "-Q" in optn_flags:
        pollist.append(1)
    if "-U" in optn_flags:
        pollist.append(2)
    if "-QU" in optn_flags:
        pollist.append(1, 2)
    if "-IQU" in optn_flags:
        pollist.append(0, 1, 2)
    if len(pollist) == 0:
        pollist = [0]
    return pollist

def get_sizes(optn_flags):
    sizes = []
    if "-small" in optn_flags:
        sizes.append(8.8)
    if "-medium" in optn_flags:
        sizes.append(12.0)
    if "-large" in optn_flags:
        sizes.append(18.0)
    if len(sizes) == 0:
        sizes = [8.8, 12.0, 18.0]
    return sizes

def get_range(optn_flags):
    min = float(get_key(optn_flags, "-min")) if "-min" in optn_flags else None
    max = float(get_key(optn_flags, "-max")) if "-max" in optn_flags else None
    if "-range" in optn_flags:
        r = float(get_key(optn_flags, "-range"))
        min = -r
        max =  r
    return min, max

def get_key(optn_index, keyword):
    return optn_index[optn_flags.index(keyword) + 1]


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
