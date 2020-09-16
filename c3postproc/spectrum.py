from matplotlib import rcParams, rc
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import numpy as np
import healpy as hp
import sys
import math
from brokenaxes import brokenaxes

import c3postproc.tools as tls
#from c3postproc.tools import *

def Spectrum(pol, long, darkmode, png, foregrounds, masks, nside):
    params = {'savefig.dpi'        : 300, # save figures to 300 dpi
              'xtick.top'          : False,
              'ytick.right'        : True, #Set to false
              'axes.spines.top'    : True, #Set to false
              'axes.spines.bottom' : True,
              'axes.spines.left'   : True,
              'axes.spines.right'  : True, #Set to false@
              'axes.grid.axis'     : 'y',
              'axes.grid'          : False,
              'ytick.major.size'   : 10,
              'ytick.minor.size'   : 5,
              'xtick.major.size'   : 10,
              'xtick.minor.size'   : 5,
              'ytick.major.width'   : 1.5,
              'ytick.minor.width'   : 1.5,
              'xtick.major.width'   : 1.5,
              'xtick.minor.width'   : 1.5,
              'axes.linewidth'      : 1.5,
              #'ytick.major.size'   : 6,
              #'ytick.minor.size'   : 3,
              #'xtick.major.size'   : 6,
              #'xtick.minor.size'   : 3,
    }
    black = 'k'
    if darkmode:
        rcParams['text.color']   = 'white'   # axes background color
        rcParams['axes.facecolor']   = 'white'   # axes background color
        rcParams['axes.edgecolor' ]  = 'white'   # axes edge color
        rcParams['axes.labelcolor']   = 'white'
        rcParams['xtick.color']    = 'white'      # color of the tick labels
        rcParams['ytick.color']    = 'white'      # color of the tick labels
        rcParams['grid.color'] =   'white'   # grid color
        rcParams['legend.facecolor']    = 'inherit'   # legend background color (when 'inherit' uses axes.facecolor)
        rcParams['legend.edgecolor']= 'white' # legend edge color (when 'inherit' uses axes.edgecolor)
        black = 'white'
    rcParams.update(params)
    
    # ---- Figure parameters ----
    if long:    
        xmin, xmax = (0.3, 4000)
        ymin, ymax = (0.05, 7e2)
        ymax15=1e3#ymax+500
        ymax2=1e8#ymax+1e8

        # Figure
        ratio = 5
        w, h = (16,8)
        fig, (ax2, ax) = plt.subplots(2,1,sharex=True,figsize=(w,h),gridspec_kw = {'height_ratios':[1, ratio]})
        aspect_ratio = w/h*1.25 # Correct for ratio
        rotdir = -1

        ax2.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.tick_params(labelbottom=False)
        ax2.xaxis.set_ticks_position('none')

        # ---- Adding broken axis lines ----
        d = .005  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax2.transAxes, color=black, clip_on=False)
        ax2.plot((-d, +d), (-d*ratio, + d*ratio), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d*ratio, +d*ratio), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
        # textsize
        freqtext = 16
        fgtext = 18

    else:
        xmin, xmax = (10, 1000)
        ymin, ymax = (0.005, 7e2)
        ymax2=ymax
        ymax15=ymax
        w, h = (12,8)
        fig, ax = plt.subplots(1,1,figsize=(w,h))
        aspect_ratio = w/h
        rotdir = 1
        #ax.set_aspect('equal', adjustable='box')
        
        freqtext = 20
        fgtext = 20

    # Spectrum parameters
    field = 1 if pol else 0
    nu  = np.logspace(np.log10(0.1),np.log10(5000),1000)
    npix = hp.nside2npix(nside)
    # Read masks
    m = np.ones((len(masks), npix))
    for i, mask in enumerate(masks):
        # Read and ud_grade mask
        if mask:
            m_temp = hp.read_map(mask, field=0, dtype=None, verbose=False)
            if hp.npix2nside(len(m_temp)) != nside:
                m[i] = hp.ud_grade(m_temp, nside)
                m[i,m[i,:]>0.5] = 1 # Set all mask values to integer    
                m[i,m[i,:]<0.5] = 0 # Set all mask values to integer   
            else:
                m[i] = m_temp


    # Get indices of smallest mask
    idx = m[np.argmax(np.sum(m, axis=1)), :] > 0.5
    print(f"Using sky fractions {np.sum(m,axis=1)/npix*100}%")
    # Looping over foregrounds and calculating spectra
    i = 0
    for fg in foregrounds.keys():
        if not fg == "Sum fg.":
            foregrounds[fg]["spectrum"] = getspec(nu*1e9, fg, foregrounds[fg]["params"], foregrounds[fg]["function"], field, nside, npix, idx, m,)
        if foregrounds[fg]["sum"]:
            if i==0:
                foregrounds["Sum fg."]["spectrum"] = foregrounds[fg]["spectrum"].copy()
            else:
                foregrounds["Sum fg."]["spectrum"] += foregrounds[fg]["spectrum"]
            i+=1

    # ---- Plotting foregrounds and labels ----
    j=0
    for label, fg in foregrounds.items(): # Plot all fgs except sumf
        if fg["gradient"]:
            k = 0
            gradient_fill(nu, fg["spectrum"][k], fill_color=fg["color"], ax=ax, alpha=0.5, linewidth=0.0,)
        else:
            if label == "Sum fg.":
                ax.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                if long:
                    ax2.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                k = 0
                try:
                    ax.loglog(nu,fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    if long:
                        ax2.loglog(nu,fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    k=1
                except:
                    pass
            else:
                if fg["spectrum"].shape[0] == 1:
                    ax.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    if long:
                        ax2.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    k = 0
                else:
                    ax.fill_between(nu,fg["spectrum"][0],fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    if long:
                        ax2.fill_between(nu,fg["spectrum"][0],fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    k = 1
    
        

        x0, idx1 = find_nearest(nu, fg["position"])
        x1, idx2 = find_nearest(nu, fg["position"]*1.2)
        y0 = fg["spectrum"][k][idx1]
        y1 = fg["spectrum"][k][idx2]
        datascaling  = np.log(xmin/xmax)/np.log(ymin/ymax)
        rotator = (datascaling/aspect_ratio)
        alpha = np.arctan(np.log(y1/y0)/np.log(x1/x0)*rotator)
        rotation =  np.rad2deg(alpha)#*rotator

        ax.annotate(label, xy=(x0,y0), xytext=(5,5), textcoords="offset pixels",  rotation=rotation, rotation_mode='anchor', fontsize=fgtext, color=fg["color"], path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))], horizontalalignment="center")
    


    if not pol:
        # ---- Plotting CO lines ----
        co10amp=50
        co21amp=co10amp*0.5
        co32amp=co10amp*0.2
    
        ax.bar(115., co10amp, color=black, width=1,)
        ax.bar(230., co21amp, color=black, width=2,)
        ax.bar(345., co32amp, color=black, width=3,)
        
        ax.text(115.*0.99,co10amp*0.33, r"CO$_{1\rightarrow 0}$", color=black, alpha=0.7, ha='right',va='top',rotation=90,fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax.text(230.*0.99,co21amp*0.33, r"CO$_{2\rightarrow 1}$", color=black, alpha=0.7, ha='right',va='top',rotation=90,fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax.text(345.*0.99,co32amp*0.33, r"CO$_{3\rightarrow 2}$", color=black, alpha=0.7, ha='right',va='top',rotation=90,fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
    # ---- Data band ranges ----
    if long:
        yscaletext = 0.75
        yscaletextup = 1.2
    else:
        yscaletextup = 1.03
        yscaletext = 0.90

    # TODO add these as args?
    haslam = True
    chipass = True
    spass = True
    cbass = True
    quijote = False
    wmap = True
    planck = True
    dirbe = True
    
    databands = {"Haslam":  {"0.408\nHaslam": {"pol": False, "show": haslam, "position": [.408, ymax2*yscaletext],  "range": [.406,.410], "color": 'purple',}},
                 "S-PASS":  {"2.303\nS-PASS":  {"pol": True, "show": spass,  "position": [2.35, ymax2*yscaletext],  "range": [2.1,2.4], "color": 'green',}},
                 "C-BASS":  {"5.0\nC-BASS":   {"pol": True, "show": spass,  "position": [5., ymax2*yscaletext],    "range": [4.,6.], "color": 'C0',}},
                 "CHI-PASS":{"1.394\nCHI-PASS":{"pol": False, "show": chipass,"position": [1.3945, ymax2*yscaletext],"range": [1.3945-0.064/2, 1.3945+0.064/2], "color": 'C5',}},
                 "QUIJOTE": {"11\nQUIJOTE":    {"pol": True, "show": quijote,"position": [11, ymax2*yscaletext],    "range":  [10.,12.], "color": 'C4',},
                             "13":             {"pol": True, "show": quijote, "position": [13, ymax2*yscaletext], "range":  [12.,14.], "color": 'C4',},
                             "17":             {"pol": True, "show": quijote, "position": [17, ymax2*yscaletext], "range":  [16.,18.], "color": 'C4',},
                             "19":             {"pol": True, "show": quijote, "position": [20, ymax2*yscaletext], "range":  [18.,21.], "color": 'C4',},
                             "31":             {"pol": True, "show": quijote, "position": [31, ymax2*yscaletext], "range":  [26.,36.], "color": 'C4',},
                             "41":             {"pol": True, "show": quijote, "position": [42, ymax2*yscaletext], "range":  [35.,47.], "color": 'C4',}},
                 "Planck":  {"30":          {"pol": True, "show": planck, "position": [27,  ymax2*yscaletext], "range": [23.9,34.5],"color": 'C1',},      # Planck 30
                             "44":          {"pol": True, "show": planck, "position": [40,  ymax2*yscaletext], "range": [39,50]    ,"color": 'C1',},      # Planck 44
                             "70":          {"pol": True, "show": planck, "position": [60,  ymax2*yscaletext], "range": [60,78]    ,"color": 'C1',},      # Planck 70
                             "100\nPlanck": {"pol": True, "show": planck, "position": [90,  ymax2*yscaletext], "range": [82,120]   ,"color": 'C1',},      # Planck 100
                             "143":         {"pol": True, "show": planck, "position": [130, ymax2*yscaletext], "range": [125,170]  ,"color": 'C1',},      # Planck 143
                             "217":         {"pol": True, "show": planck, "position": [195, ymax2*yscaletext], "range": [180,265]  ,"color": 'C1',},      # Planck 217
                             "353":         {"pol": True, "show": planck, "position": [320, ymax2*yscaletext], "range": [300,430]  ,"color": 'C1',},      # Planck 353
                             "545":         {"pol": False, "show": planck, "position": [490, ymax2*yscaletext], "range": [450,650]  ,"color": 'C1',},      # Planck 545
                             "857":         {"pol": False, "show": planck, "position": [730, ymax2*yscaletext], "range": [700,1020] ,"color": 'C1',}},      # Planck 857
                 "DIRBE":   {"1250\nDIRBE":  {"pol": False, "show": dirbe, "position": [1000, ymax2*yscaletext], "range": [1000,1540] , "color": 'C3',},     # DIRBE 1250
                             "2140":         {"pol": False, "show": dirbe, "position": [1750, ymax2*yscaletext], "range": [1780,2500] , "color": 'C3',},     # DIRBE 2140
                             "3000":         {"pol": False, "show": dirbe, "position": [2500, ymax2*yscaletext], "range": [2600,3500] , "color": 'C3',}},     # DIRBE 3000
                 "WMAP":    {"WMAP\nK": {"pol": True, "show": wmap, "position": [21.8, ymin*yscaletextup], "range": [21,25.5], "color": 'C9',}, 
                             "Ka":      {"pol": True, "show": wmap, "position": [31.5, ymin*yscaletextup], "range": [30,37], "color": 'C9',},
                             "Q":       {"pol": True, "show": wmap, "position": [39.,  ymin*yscaletextup], "range": [38,45], "color": 'C9',}, 
                             "V":       {"pol": True, "show": wmap, "position": [58.,  ymin*yscaletextup], "range": [54,68], "color": 'C9',}, 
                             "W":       {"pol": True, "show": wmap, "position": [90.,  ymin*yscaletextup], "range": [84,106], "color": 'C9',}}, 
    }
    
    # Set databands from dictonary
    for experiment, bands in databands.items():
        for label, band in bands.items():
            if band["show"]:
                if pol and not band["pol"]:
                    continue # Skip non-polarization bands
                if band["position"][0]>xmax or band["position"][0]<xmin:
                    continue # Skip databands outside range
                va = "bottom" if experiment == "WMAP" else "top" # VA for WMAP on bottom
                ha = "left" if experiment in ["Planck", "WMAP", "DIRBE"] else "center"
                ax.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=0, label=experiment)
                if long:
                    ax2.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=0, label=experiment)
                    if experiment == "WMAP":
                        ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1,-1))])
                    else:
                        ax2.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1,-1))])
                else:
                    ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1,-1))])

    # ---- Axis stuff ----
    lsize=20
    ax.set(xscale='log', yscale='log', ylim=(ymin, ymax), xlim=(xmin,xmax), xticks=[10,30,100,300,1000,])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='both', which='major', labelsize=lsize, direction='in')
    ax.tick_params(which="both",direction="in")
    if long:
        ax2.set(xscale='log', yscale='log', ylim=(ymax15, ymax2), xlim=(xmin,xmax), yticks=[1e4,1e6,])
        ax.set_xticks([1,3,10,30,100,300,1000,3000])
        ax2.tick_params(axis='both', which='major', labelsize=lsize, direction='in')
        ax2.tick_params(which="both",direction="in")

    # Axis labels
    plt.ylabel(r"RMS brightness temperature [$\mu$K]",fontsize=lsize)
    plt.xlabel(r"Frequency [GHz]",fontsize=lsize)

    #ax.legend(loc=6,prop={'size': 20}, frameon=False)

    # ---- Plotting ----
    plt.tight_layout(h_pad=0.3)
    filename = "spectrum"
    filename += "_pol" if pol else ""
    filename += "_long" if long else ""
    filename += "_darkmode" if darkmode else ""
    filename += ".png" if png else ".pdf"
    print("Plotting {}".format(filename))
    plt.savefig(filename, bbox_inches='tight',  pad_inches=0.02, transparent=True)


	
def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon

    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# This function calculates the intensity spectra
# Alternative 1 uses 2 masks to calculate spatial variations
# Alternative 2 uses only scalar values
def getspec(nu, fg, params, function, field, nside, npix, idx, m):
    val = []
    # Alternative 1
    if any([str(x).endswith(".fits") for x in params]):
        if fg == "Spinning Dust":
            from pathlib import Path
            ame_template = Path(__file__).parent / "spdust2_cnm.dat"
            fnu, f_ = np.loadtxt(ame_template, unpack=True)
            fnu *= 1e9
            field = 0

        temp = []
        nsides = []
        # Read all maps and record nsides
        
        for i, p in enumerate(params):
            if str(p).endswith(".fits"):
                if field==1 and i==0: # If polarization amplitude map
                    s1 = hp.read_map(p, field=1, dtype=None, verbose=False)
                    s2 = hp.read_map(p, field=2, dtype=None, verbose=False)
                    p = np.sqrt(s1**2+s2**2)
                else:
                    p = hp.read_map(p, field=field, dtype=None, verbose=False)
                nsides.append(hp.npix2nside(len(p)))
            else:
                nsides.append(0)
            temp.append(p)  


        # Create dataset and convert to same resolution
        params = np.zeros(( len(params), npix ))
        for i, t in enumerate(temp):
            if nsides[i] == 0:
                params[i,:] = t
            elif nsides[i] != nside:
                params[i,:] = hp.ud_grade(t, nside)

        # Only calculate outside masked region    
        N = 1000
        map_ = np.zeros((N, npix))

        for i, nu_ in enumerate(tqdm(nu, desc = fg, ncols=80)):
            if fg == "Spinning Dust":
                map_[i, idx] = getattr(tls, function)(nu_, *params[:,idx], fnu, f_) #fgs.fg(nu, *params[pix])
            else:
                map_[i, idx] = getattr(tls, function)(nu_, *params[:,idx]) #fgs.fg(nu, *params[pix])

        # Apply mask to all frequency points
        # calculate mean 
        rmss = []
        for i in range(2):
            n = np.sum(m[i])            
            masked = hp.ma(map_)
            masked.mask = np.logical_not(m[i])
            mono = masked.mean(axis=1)
            masked -= mono.reshape(-1,1)
        
            rms = np.sqrt( ( (masked**2).sum(axis=1) ) /n)
            val.append(rms)

        vals = np.sort(np.array(val), axis=0) 
    else:
        # Alternative 2
        val = getattr(tls, function)(nu, *params) #fgs.fg(nu, *params))
        #vals = np.stack((val, val),)
        vals = val.reshape(1,-1)
    return vals
