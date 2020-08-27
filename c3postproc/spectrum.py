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

def Spectrum(pol, long, lowfreq, darkmode, png, foregrounds, masks):
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
    def getspec(nu, fg, params, masks, field):
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
            nside_max = 4 #2 #np.max(nsides)
            npix = hp.nside2npix(nside_max)

            # Create dataset and convert to same resolution
            params = np.zeros(( len(params), npix ))
            for i, t in enumerate(temp):
                if nsides[i] == 0:
                    params[i,:] = t
                elif nsides[i] != nside_max:
                    params[i,:] = hp.ud_grade(t, nside_max)

            # Calculate spectra USE JIT?!
            map_ = np.zeros((N, npix))
            print(f"Calculating {fg} values over all pixels and freqs")
            for i, nu_ in enumerate(tqdm(nu, desc = "Freqs", ncols=80)):
                for pix in trange(npix, desc = "pixel", leave=False, ncols=80):
                    #print(f"params of {fg}: {params[:,pix]}")
                    if fg == "sdust":
                        map_[i, pix] = getattr(tls.fgs, fg)(nu_, *params[:,pix], fnu, f_) #fgs.fg(nu, *params[pix])
                    else:
                        map_[i, pix] = getattr(tls.fgs, fg)(nu_, *params[:,pix]) #fgs.fg(nu, *params[pix])
                       
            # Apply mask to all frequency points
            # calculate mean 
            for i, mask in enumerate(masks):
                # Read and ud_grade mask
                if not mask: 
                    continue
                m = hp.read_map(mask, field=field, dtype=None, verbose=False)
                if hp.npix2nside(len(m)) != nside_max:
                    m = hp.ud_grade(m, nside_max)
                m[m>0.5] = 1 # Set all mask values to integer    
                m[m<0.5] = 0 # Set all mask values to integer    

                n = np.sum(m) # Total valid pixels
                masked = map_*m.reshape(1,-1) # Mask data
                mu = np.sum(masked, axis=1)/n # Average for each freq point
                mystery = np.sqrt((np.sum(masked-mu.reshape(-1,1), axis=1)**2)/n)
                val.append(mystery)
                       
            vals = np.sort(np.array(val), axis=0)
        # Alternative 2
        else:
            val = getattr(tls.fgs, fg)(nu, *params) #fgs.fg(nu, *params))
            vals = np.stack((val, val),)
        return vals

    fgs = []
    field = 1 if pol else 0
    N = 1000
    nu  = np.logspace(np.log10(0.1),np.log10(5000),N)
    for fg in foregrounds.keys():
        fgs.append(getspec(nu*1e9, fg, foregrounds[fg], masks, field) )

    #for i in range(len(fgs)):
    #    print(fgs[i])

    """
    OLD METHOD
    
    ## FIND MIN MAX
    def findminmax(nu, function, numparams, range1=None,range2=None,range3=None, range4=None):
        vals = np.zeros((2, len(nu)))
        val = []
        for i in range1:
            if numparams >= 2:
                for j in range2:
                    if numparams >= 3:
                        for k in range3:
                            if numparams == 4:
                                for l in range4:                                
                                    val.append(fgs.function(nu, i, j,k,l))
                            else:
                                val.append(fgs.function(nu, i, j, k))
                    else:
                        val.append(fgs.function(nu, i, j))
            else:
                val.append(fgs.function(nu, i))
    
        val = np.array(val)
        vals[0,:] = np.min(val,axis=0)
        vals[1,:] = np.max(val,axis=0)
        return vals
    
    def mb(range, n=5):
        return np.linspace(range[0]-range[1],range[0]+range[1], 10)
    
    cmb_range = [67,1]

    te_range = [7000, 11]
    EM_range = [30,5]#[13, 1]
    
    ame1_a = [92,1]#[92,118]
    ame2_a = [17,1]#[17,22]
    ame_nu = [19,1] #[19,1]
    
    ame1_a = [50,5]#[92,118]
    ame2_a = [50,5]#[17,22]
    ame_nu = [22.8e9, 1e9] #[19,1]
    
    dust_a = [163,30] #[163,228]
    dust_t = [21,2]
    dust_b = [1.51, 0.05]
    
    sync_a = [20*1e6,1*1e6] #[20,15]
    
    cmb_pol = [0.67,0.03]
    sync_pol = [12, 1] #[12,9]
    dust_pol = [8,1] #[8,10]
    
    if not pol:
    	CMB = findminmax(nu*1e9, cmb, numparams=1, range1=mb(cmb_range)) # 70
    	FF    = findminmax(nu*1e9, ff,numparams=2, range1=mb(EM_range),range2=mb(te_range)) #, 30., 7000.)
    	SYNC  = findminmax(nu*1e9, sync, numparams=2, range1=mb(sync_a), range2=[1]) # 30.*1e6,1.)
    	TDUST = findminmax(nu*1e9, tdust, numparams=4, range1=mb(dust_a), range2=mb(dust_b), range3=mb(dust_t), range4=[545.]) # 163, 1.6,21.)
    	
    	SDUST1 = findminmax(1.5*nu*1e9, sdust, numparams=2, range1=mb(ame1_a),range2=[41e9])#,  (1.5*nu*1e9, 50, 41e9)+
    	SDUST2 = findminmax(0.9*nu*1e9, sdust, numparams=2, range1=mb(ame2_a), range2=mb(ame_nu))#,  sdust(0.9*nu*1e9, 50, 22.8e9)
    	SDUST = SDUST1+SDUST2
    	
    	# REFERANGE?
    	
    	CMB   = cmb(  nu*1e9, 70)
    	FF    = ff(   nu*1e9, 30., 7000.)
    	SYNC  = sync( nu*1e9, 30.*1e6,1.)
    	SDUST = sdust(1.5*nu*1e9, 50, 41e9)+sdust(0.9*nu*1e9, 50, 22.8e9)
        T = tdust(nu*1e9, 163, 1.6,21.)    
    else:
        CMB = findminmax(nu*1e9, cmb, numparams=1, range1=mb(cmb_pol)) # 70
        SYNC  = findminmax(nu*1e9, sync, numparams=3, range1=mb(sync_pol), range2=[1], range3=[30.]) # 30.*1e6,1.)
        TDUST = findminmax(nu*1e9, tdust, numparams=4, range1=mb(dust_pol), range2=mb(dust_b), range3=mb(dust_t),range4=[353.]) # 163, 1.6,21.)
    
    """

    # calculate sky model
    """
    Functions should take numbers
    A, B, T
    if A B and T are scalar, just insert, and calculate returning scalar
    if not:
            read maps, rescale maps to the lowest resolution of inputs
            calculate spectrum in each pixel for N frequency points (Set in input?)
            return masked average (input optional list of masks?)
    """

    
    sumf = np.sum(fgs, axis=0) # SYNC+TDUST+SDUST*0.01
    fgs.append(sumf)
    if pol:	
        #CMB   = cmb(  nu*1e9, 0.67)
        #SYNC  = sync( nu*1e9, 12,1., nuref=30.)
        #TDUST = tdust(nu*1e9, 8, 1.51,21.,nuref=353. )
        

        #fgs=[CMB,SYNC,TDUST,SDUST*0.01]
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
    
    
    ax.loglog(nu,sumf[0], "--", linewidth=2, color=black, alpha=0.7)
    ax.loglog(nu,sumf[1], "--", linewidth=2, color=black, alpha=0.7)
    ax2.loglog(nu,sumf[0], "--", linewidth=2, color=black, alpha=0.7)
    ax2.loglog(nu,sumf[1], "--", linewidth=2, color=black, alpha=0.7)
    # ---- Plotting foregrounds and labels ----
    j=0
    for i in range(len(fgs)):
        linestyle = "dotted" if pol and i == 3 else "solid" # Set upper boundry
        ax.fill_between(nu, fgs[i][0], fgs[i][1], color=col[i], linestyle=linestyle)
        ax2.fill_between(nu, fgs[i][0], fgs[i][1], color=col[i], linestyle=linestyle)
    
        #ax.loglog(nu,fgs[i], linewidth=4,color=col[i])
        #ax2.loglog(nu,fgs[i], linewidth=4,color=col[i])
        #ax.text(nu[idx[i]], fgs[i][1,idx[i]]+scale[i], label[i], rotation=rot[i], color=col[i],fontsize=fgtext,  path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
    
    # ---- Plotting sum of all foregrounds ----        
    #ax.text(nu[idx[-1]], fgs[-1][1,idx[-1]]+scale[-1], label[-1], rotation=rot[-1], color=black, fontsize=fgtext, alpha=0.7,  path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))])
    
    
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
    
    plt.ylabel(r"Brightness temperature [$\mu$K]",fontsize=labelsize)
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
	
