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
from pkg_resources import resource_filename


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
        from pathlib import Path
        #sys.exit()
        color = Path(__file__).parent / 'parchment1.dat'
        print(color)

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

"""
def get_cmbcolormap():
   0 0 255
   0 2 255
   0 5 255
   0 8 255
   0 10 255
   0 13 255
   0 16 255
   0 18 255
   0 21 255
   0 24 255
   0 26 255
   0 29 255
   0 32 255
   0 34 255
   0 37 255
   0 40 255
   0 42 255
   0 45 255
   0 48 255
   0 50 255
   0 53 255
   0 56 255
   0 58 255
   0 61 255
   0 64 255
   0 66 255
   0 69 255
   0 72 255
   0 74 255
   0 77 255
   0 80 255
   0 82 255
   0 85 255
   0 88 255
   0 90 255
   0 93 255
   0 96 255
   0 98 255
   0 101 255
   0 104 255
   0 106 255
   0 109 255
   0 112 255
   0 114 255
   0 117 255
   0 119 255
   0 122 255
   0 124 255
   0 127 255
   0 129 255
   0 132 255
   0 134 255
   0 137 255
   0 139 255
   0 142 255
   0 144 255
   0 147 255
   0 150 255
   0 152 255
   0 155 255
   0 157 255
   0 160 255
   0 162 255
   0 165 255
   0 167 255
   0 170 255
   0 172 255
   0 175 255
   0 177 255
   0 180 255
   0 182 255
   0 185 255
   0 188 255
   0 190 255
   0 193 255
   0 195 255
   0 198 255
   0 200 255
   0 203 255
   0 205 255
   0 208 255
   0 210 255
   0 213 255
   0 215 255
   0 218 255
   0 221 255
   6 221 254
  12 221 253
  18 222 252
  24 222 251
  30 222 250
  36 223 249
  42 223 248
  48 224 247
  54 224 246
  60 224 245
  66 225 245
  72 225 244
  78 225 243
  85 226 242
  91 226 241
  97 227 240
 103 227 239
 109 227 238
 115 228 237
 121 228 236
 127 229 236
 133 229 235
 139 229 234
 145 230 233
 151 230 232
 157 230 231
 163 231 230
 170 231 229
 176 232 228
 182 232 227
 188 232 226
 194 233 226
 200 233 225
 206 233 224
 212 234 223
 218 234 222
 224 235 221
 230 235 220
 236 235 219
 242 236 218
 248 236 217
 255 237 217
 255 235 211
 255 234 206
 255 233 201
 255 231 196
 255 230 191
 255 229 186
 255 227 181
 255 226 176
 255 225 171
 255 223 166
 255 222 161
 255 221 156
 255 219 151
 255 218 146
 255 217 141
 255 215 136
 255 214 131
 255 213 126
 255 211 121
 255 210 116
 255 209 111
 255 207 105
 255 206 100
 255 205 95
 255 203 90
 255 202 85
 255 201 80
 255 199 75
 255 198 70
 255 197 65
 255 195 60
 255 194 55
 255 193 50
 255 191 45
 255 190 40
 255 189 35
 255 187 30
 255 186 25
 255 185 20
 255 183 15
 255 182 10
 255 181 5
 255 180 0
 255 177 0
 255 175 0
 255 172 0
 255 170 0
 255 167 0
 255 165 0
 255 162 0
 255 160 0
 255 157 0
 255 155 0
 255 152 0
 255 150 0
 255 147 0
 255 145 0
 255 142 0
 255 140 0
 255 137 0
 255 135 0
 255 132 0
 255 130 0
 255 127 0
 255 125 0
 255 122 0
 255 120 0
 255 117 0
 255 115 0
 255 112 0
 255 110 0
 255 107 0
 255 105 0
 255 102 0
 255 100 0
 255 97 0
 255 95 0
 255 92 0
 255 90 0
 255 87 0
 255 85 0
 255 82 0
 255 80 0
 255 77 0
 255 75 0
 251 73 0
 247 71 0
 244 69 0
 240 68 0
 236 66 0
 233 64 0
 229 62 0
 226 61 0
 222 59 0
 218 57 0
 215 55 0
 211 54 0
 208 52 0
 204 50 0
 200 48 0
 197 47 0
 193 45 0
 190 43 0
 186 41 0
 182 40 0
 179 38 0
 175 36 0
 172 34 0
 168 33 0
 164 31 0
 161 29 0
 157 27 0
 154 26 0
 150 24 0
 146 22 0
 143 20 0
 139 19 0
 136 17 0
 132 15 0
 128 13 0
 125 12 0
 121 10 0
 118 8 0
 114 6 0
 110 5 0
 107 3 0
 103 1 0
 100 0 0
 """