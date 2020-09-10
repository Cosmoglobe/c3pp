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

def Spectrum(pol, long, lowfreq, darkmode, png, foregrounds, masks, nside):
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
    baralpha= 0.3
    ratio=5
    if long:
        xmin=0.3
        xmax=4000
        ymin=0.05
        ymax=7e2
        ymax2=1e8#ymax+1e8
        ymax15=1000#ymax+500
    
        fig, (ax2, ax) = plt.subplots(2,1,sharex=True,figsize=(16,8),gridspec_kw = {'height_ratios':[1, ratio]})
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
    
        freqtext = 16
        fgtext = 18
        labelsize = 20
        ticksize = 20
    
    else:
        xmin=10
        xmax=1000
        ymin=0.05
        ymax=7e2
        ymax2=ymax
        ymax15=ymax
    
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        ax2 = ax
    
        freqtext = 20
        fgtext = 20
        labelsize = 20
        ticksize = 20

    # This function calculates the intensity spectra
    # Alternative 1 uses 2 masks to calculate spatial variations
    # Alternative 2 uses only scalar values
    def getspec(nu, fg, params, field, nside, npix, idx, m):
        val = []
        # Alternative 1
        if any([str(x).endswith(".fits") for x in params]):
            if fg == "sdust":
                from pathlib import Path
                ame_template = Path(__file__).parent / "spdust2_cnm.dat"
                fnu, f_ = np.loadtxt(ame_template, unpack=True)
                fnu *= 1e9
                field = 0

            temp = []
            nsides = []
            # Read all maps and record nsides
            for p in params:
                if str(p).endswith(".fits"):
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
            map_ = np.zeros((N, npix))

            for i, nu_ in enumerate(tqdm(nu, desc = fg, ncols=80)):
                if fg == "sdust":
                    map_[i, idx] = getattr(tls.fgs, fg)(nu_, *params[:,idx], fnu, f_) #fgs.fg(nu, *params[pix])
                else:
                    map_[i, idx] = getattr(tls.fgs, fg)(nu_, *params[:,idx]) #fgs.fg(nu, *params[pix])
            """
            if fg == "sdust":
                map_[:, idx] = getattr(tls.fgs, fg)(nu, *params[:,idx], fnu, f_) #fgs.fg(nu, *params[pix])
            else:
                map_[:, idx] = getattr(tls.fgs, fg)(nu, *params[:,idx]) #fgs.fg(nu, *params[pix])
            """
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
            val = getattr(tls.fgs, fg)(nu, *params) #fgs.fg(nu, *params))
            #vals = np.stack((val, val),)
            vals = val.reshape(1,-1)
        return vals

    # Spectrum parameters
    fgs = []
    field = 1 if pol else 0
    N = 1000
    nu  = np.logspace(np.log10(0.1),np.log10(5000),N)
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

    #hp.write_map("mask0.fits", m[0])
    #hp.write_map("mask1.fits", m[1])
    # Get indices of smallest mask
    idx = m[np.argmax(np.sum(m, axis=1)), :] > 0.5
    print(f"Using sky fractions {np.sum(m,axis=1)/npix*100}%")
    # Looping over foregrounds and calculating spectra
    i = 0
    label = []
    for fg in foregrounds.keys():
        spectrum = getspec(nu*1e9, fg, foregrounds[fg], field, nside, npix, idx, m,)
        fgs.append(spectrum)
        label.append(fg)
        if fg != "cmb":
            if i==0:
                sumf = spectrum.copy()
            else:
                sumf += spectrum
            i+=1

    #fgs.append(sumf)
    label.append("Sum Fg.")

    if pol:	
        col=["C9","C2","C3","C1","C7"]
        label=["CMB", "Synchrotron","Thermal Dust", "Spinning Dust", "Sum fg."]
        if long:
            rot=[-25, -51, 22, 10, -10] 
            idx=[70, -120,  115, -215, -15]
            scale=[0.05, 0, 13.5, 1.5, 3]
        else:
            rot=[-20, -45, 18, -15, -10]
            idx=[70, -120,  115, -145, -15]
            scale=[0.05, 0, 10, 0.5, 3]
    
    else:
        col=["C9","C0","C2","C1","C3","C7"]
        label=["CMB","Free-Free","Synchrotron","Spinning Dust","Thermal Dust", "Sum fg."] 
        if long:
            rot=[-8, -40, -50, -73, 13, -45]
            idx=[17, 50, 50, -10, 160, -90]
            scale=[5,0,0,0,200,300]
        else:  
            rot=[-8, -35, -46, -70, 13, -40] 
            idx=[17, 60, 58, -10, 160, -90]
            scale=[5,0,0,0,150,300]
    
        if lowfreq:
            rot=[-8, -37, -47, -70, 13, -43]
    scale=[0,0,0,0,0,0,0]
    idxshift = 600
    idx = [x + idxshift for x in idx]
    
    
    haslam = True
    chipass = True
    spass = True
    cbass = True
    quijote = False
    wmap = True
    planck = True
    dirbe = True
    
    
    # ---- Foreground plotting parameters ----
    
    #scale=[5,105,195] # Scaling CMB, thermal dust and sum up and down
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax.loglog(nu,sumf[0], "--", linewidth=2, color=black, alpha=0.7, label=label[-1])
    ax2.loglog(nu,sumf[0], "--", linewidth=2, color=black, alpha=0.7)
    try:
        ax.loglog(nu,sumf[1], "--", linewidth=2, color=black, alpha=0.7)
        ax2.loglog(nu,sumf[1], "--", linewidth=2, color=black, alpha=0.7)
    except:
        pass
    # ---- Plotting foregrounds and labels ----
    j=0
    for i, fg in enumerate(fgs): # Plot all fgs except sumf
        linestyle = "dotted" if pol and i == 3 else "solid" # Set upper boundry
        if fgs[i].shape[0] == 1:
            ax.plot(nu, fg[0], color=col[i], linestyle=linestyle, linewidth=4,label=label[i])
            ax2.plot(nu, fg[0], color=col[i], linestyle=linestyle, linewidth=4,)
            k=0
        else:
            ax.fill_between(nu, fg[0], fg[1], color=col[i], linestyle=linestyle,label=label[i])
            ax2.fill_between(nu, fg[0], fg[1], color=col[i], linestyle=linestyle)
            k=1
        #ax.loglog(nu,fgs[i], linewidth=4,color=col[i])
        #ax2.loglog(nu,fgs[i], linewidth=4,color=col[i])
        ax.text(nu[idx[i]], fg[k][idx[i]]+scale[i], label[i], rotation=rot[i], color=col[i],fontsize=fgtext,  path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        
    # ---- Plotting sum of all foregrounds ----        
    ax.text(nu[idx[-1]], sumf[1][idx[-1]]+scale[-1], label[-1], rotation=rot[-1], color=black, fontsize=fgtext, alpha=0.7,  path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    #ax.text(10, find_nearest(fgs[-1], 10), label[-1], rotation=rot[-1], color=col[-1],fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
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
    band_range1   = [.406,.410]      # Haslam?
    band_range2   = [2.1,2.4]      # Spass?
    band_range3   = [21,25.5]        # WMAP K
    band_range4   = [30,37]          # WMAP Ka
    band_range5   = [38,45]          # WMAP Q
    band_range6   = [54,68]          # WMAP V
    band_range7   = [84,106]         # WMAP W
    band_range8   = [23.9,34.5]      # Planck 30
    band_range9   = [39,50]          # Planck 44
    band_range10  = [60,78]          # Planck 70
    band_range11  = [82,120]         # Planck 100
    band_range12  = [125,170]        # Planck 143
    band_range13  = [180,265]        # Planck 217
    band_range14  = [300,430]        # Planck 353
    band_range15  = [450,650]        # Planck 545
    band_range16  = [700,1020]       # Planck 857
    band_range17  = [1000,1540]      # DIRBE 1250
    band_range18  = [1780,2500]      # DIRBE 2140
    band_range19  = [2600,3500]      # DIRBE 3000
    band_range20  = [4.,6.]      # C-BASS
    band_range21  = [10.,12.]      # QUIJOTE
    band_range22  = [12.,14.]      # QUIJOTE
    band_range23  = [16.,18.]      # QUIJOTE
    band_range24  = [18.,21.]      # QUIJOTE
    band_range25  = [26.,36.]      # QUIJOTE
    band_range26  = [35.,47.]      # QUIJOTE
    band_range27  = [1.3945-0.064/2,1.3945+0.064/2]  #CHIPASS
    
    if long:
        yscaletext = 0.80
        yscaletextup = 1.2
    else:
        yscaletextup = 1.03
        yscaletext = 0.97
    
    # ---- Plotting single data ----
    if long:
        if haslam and not pol:
            ax2.text(np.mean(band_range1),ymax2*yscaletext,"0.408\nHaslam",color='purple',va='top',horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
            ax.axvspan(band_range1[0],band_range1[1],color='purple',alpha=baralpha, zorder=0,label="Haslam")
            ax2.axvspan(band_range1[0],band_range1[1],color='purple',alpha=baralpha, zorder=0)
    
        if spass:
            ax2.text(np.mean(band_range2)+.1 ,ymax2*yscaletext,"2.303\nS-PASS",color='green',va='top',horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
            ax.axvspan(band_range2[0],band_range2[1],color='green',alpha=baralpha, zorder=0, label="S-PASS")
            ax2.axvspan(band_range2[0],band_range2[1],color='green',alpha=baralpha, zorder=0)
    
        if cbass:
            ax2.text(np.mean(band_range20),ymax2*yscaletext,"5.0\nC-BASS",color='C0',va='top',horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
            ax.axvspan(band_range20[0],band_range20[1],color='C0',alpha=baralpha, zorder=0, label="C-BASS")
            ax2.axvspan(band_range20[0],band_range20[1],color='C0',alpha=baralpha, zorder=0)
    
        if chipass and not pol:
            ax2.text(np.mean(band_range27)-0.1,ymax2*yscaletext,"1.394\nCHIPASS",color='C5', va='top',horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
            ax.axvspan(band_range27[0],band_range27[1],color='C5',alpha=baralpha, zorder=0,label='CHIPASS')
            ax2.axvspan(band_range27[0],band_range27[1],color='C5',alpha=baralpha, zorder=0)
    
    
    # ---- Plotting QUIJOTE ----
    if quijote:
        ax2.text(11,ymax2*yscaletext,"11\nQUIJOTE",  color='C4', va='top',alpha=1,horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(13,ymax2*yscaletext,"13",  color='C4', va='top',alpha=1,horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(17,ymax2*yscaletext,"17",  color='C4', va='top',alpha=1,horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(19+1,ymax2*yscaletext,"19", color='C4', va='top',alpha=1,horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(31,ymax2*yscaletext,"31",color='C4', va='top',alpha=1,horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(41+1,ymax2*yscaletext,"41",color='C4', va='top',alpha=1,horizontalalignment='center', size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
        ax.axvspan(band_range21[0], band_range21[1], color='C4',alpha=baralpha, zorder=0, label="QUIJOTE")
        ax.axvspan(band_range22[0], band_range22[1], color='C4',alpha=baralpha, zorder=0)
        ax.axvspan(band_range23[0], band_range23[1], color='C4',alpha=baralpha, zorder=0)
        ax.axvspan(band_range24[0], band_range24[1], color='C4',alpha=baralpha, zorder=0)
        ax.axvspan(band_range25[0], band_range25[1], color='C4',alpha=baralpha, zorder=0)
        ax.axvspan(band_range26[0], band_range26[1], color='C4',alpha=baralpha, zorder=0)
        if long:
            ax2.axvspan(band_range21[0],band_range21[1], color='C4',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range22[0],band_range22[1], color='C4',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range23[0],band_range23[1], color='C4',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range24[0],band_range24[1], color='C4',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range25[0],band_range25[1], color='C4',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range26[0],band_range26[1], color='C4',alpha=baralpha, zorder=0)
    
    # ---- Plotting Planck ----
    if planck:
        ax2.text(27,ymax2*yscaletext,"30",           color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(44-4,ymax2*yscaletext,"44",           color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(64-4,  ymax2*yscaletext,"70",           color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(90,ymax2*yscaletext,"100\nPlanck", color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(130, ymax2*yscaletext,"143",          color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(195, ymax2*yscaletext,"217",          color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(320, ymax2*yscaletext,"353",          color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
        ax.axvspan(band_range8[0] ,band_range8[1], color='C1',alpha=baralpha, zorder=0, label="Planck")
        ax.axvspan(band_range9[0] ,band_range9[1], color='C1',alpha=baralpha, zorder=0)
        ax.axvspan(band_range10[0],band_range10[1],color='C1',alpha=baralpha, zorder=0)
        ax.axvspan(band_range11[0],band_range11[1],color='C1',alpha=baralpha, zorder=0)
        ax.axvspan(band_range12[0],band_range12[1],color='C1',alpha=baralpha, zorder=0)
        ax.axvspan(band_range13[0],band_range13[1],color='C1',alpha=baralpha, zorder=0)
        ax.axvspan(band_range14[0],band_range14[1],color='C1',alpha=baralpha, zorder=0)
    
        if long:
            ax2.axvspan(band_range8[0] ,band_range8[1], color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range9[0] ,band_range9[1], color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range10[0],band_range10[1],color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range11[0],band_range11[1],color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range12[0],band_range12[1],color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range13[0],band_range13[1],color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range14[0],band_range14[1],color='C1',alpha=baralpha, zorder=0)
    
        if not pol:
            ax2.text(490, ymax2*yscaletext,"545", color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
            ax2.text(730, ymax2*yscaletext,"857", color='C1', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
            ax.axvspan(band_range15[0],band_range15[1],color='C1',alpha=baralpha, zorder=0)
            ax.axvspan(band_range16[0],band_range16[1],color='C1',alpha=baralpha, zorder=0)
            if long:
                ax2.axvspan(band_range15[0],band_range15[1],color='C1',alpha=baralpha, zorder=0)
                ax2.axvspan(band_range16[0],band_range16[1],color='C1',alpha=baralpha, zorder=0)
    
    # ---- Plotting WMAP ----
    if wmap:
        ax.text(22.8  -1, ymin*yscaletextup,"WMAP\nK", color='C9' ,va='bottom',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax.text(31.5   ,   ymin*yscaletextup,"Ka ",        color='C9', va='bottom',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax.text(39.  ,     ymin*yscaletextup,"Q",         color='C9' ,va='bottom',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax.text(58.    ,   ymin*yscaletextup,"V",         color='C9' ,va='bottom',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax.text(90.   ,  ymin*yscaletextup,"W",         color='C9' ,va='bottom',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
        ax.axvspan(band_range3[0],band_range3[1],color='C9',alpha=baralpha, zorder=0,label='WMAP')
        ax.axvspan(band_range4[0],band_range4[1],color='C9',alpha=baralpha, zorder=0)
        ax.axvspan(band_range5[0],band_range5[1],color='C9',alpha=baralpha, zorder=0)
        ax.axvspan(band_range6[0],band_range6[1],color='C9',alpha=baralpha, zorder=0)
        ax.axvspan(band_range7[0],band_range7[1],color='C9',alpha=baralpha, zorder=0)
        if long:
            ax2.axvspan(band_range3[0],band_range3[1],color='C9',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range4[0],band_range4[1],color='C9',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range5[0],band_range5[1],color='C9',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range6[0],band_range6[1],color='C9',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range7[0],band_range7[1],color='C9',alpha=baralpha, zorder=0)
    
    # ---- Plotting DIRBE ----
    if dirbe and not pol and long and not lowfreq:
        ax2.text(1000  ,ymax2*yscaletext,"1249\nDIRBE",color='C3', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(1750  ,ymax2*yscaletext,"2141",color='C3', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
        ax2.text(2500  ,ymax2*yscaletext,"2998",color='C3', va='top',alpha=1, size = freqtext, path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
        ax.axvspan(band_range17[0],band_range17[1],color='C3',alpha=baralpha, zorder=0,label='DIRBE')
        ax.axvspan(band_range18[0],band_range18[1],color='C3',alpha=baralpha, zorder=0)
        ax.axvspan(band_range19[0],band_range19[1],color='C3',alpha=baralpha, zorder=0)
    
        ax2.axvspan(band_range17[0],band_range17[1],color='C3',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range18[0],band_range18[1],color='C3',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range19[0],band_range19[1],color='C3',alpha=baralpha, zorder=0)
    
    # ---- Axis label stuff ----
    #ax.set_xticks(np.append(ax.get_xticks(),[3,30,300,3000])) #???
    #ax.set_xticklabels(np.append(ax.get_xticks(),300))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='both', which='major', labelsize=ticksize, direction='in')
    ax.tick_params(which="both",direction="in")
    
    ax2.tick_params(axis='both', which='major', labelsize=ticksize, direction='in')
    ax2.tick_params(which="both",direction="in")
    
    plt.ylabel(r"RMS brightness temperature [$\mu$K]",fontsize=labelsize)
    plt.xlabel(r"Frequency [GHz]",fontsize=labelsize)
    
    if lowfreq:
        xmax = 1000
    if long:
        ax2.set_ylim(ymax15,ymax2)
        ax2.set_xlim(xmin,xmax)
    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    #ax.legend(loc=6,prop={'size': 20}, frameon=False)

    # ---- Plotting ----
    plt.tight_layout(h_pad=0.3)
    filename = "spectrum"
    filename += "_pol" if pol else ""
    filename += "_long" if long else ""
    filename += "_lowfreq" if lowfreq else ""
    filename += "_darkmode" if darkmode else ""
    filename += ".png" if png else ".pdf"
    print("Plotting {}".format(filename))
    plt.savefig(filename, bbox_inches='tight',  pad_inches=0.02, transparent=True)
    #plt.show()
	
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
