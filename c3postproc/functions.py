import math
import sys
import os
import time
import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd

def stddev(flags):
    tag = str(flags[0])
    sig = flags[1]       # T Q or U or (1, 2, 3)
    min = int(flags[2])
    max = int(flags[3])
    sig = flags[4] 
    outname = flags[-1]

    if str(sig) == "I":
        sig = 1
    elif str(sig) == "Q":
        sig = 2
    elif str(sig) == "U":
        sig = 3
    else:
        sig = int(sig)
    l = max-min
    maps = []
    for i in range(min,max+1):
        maps.append( hp.read_map(tag+"_c0001_k"+str(i).zfill(6)+".fits", field=sig) )
    maps = np.array(maps)

    outmap = np.std(maps, axis=0)
    hp.write_map(outname, outmap, overwrite=True)

def mean(flags):
    tag = str(flags[0])
    sig = flags[1]       # T Q or U or (1, 2, 3)
    min = int(flags[2])
    max = int(flags[3])
    sig = flags[4] 
    outname = flags[-1]
    if str(sig) == "I":
        sig = 1
    elif str(sig) == "Q":
        sig = 2
    elif str(sig) == "U":
        sig = 3
    else:
        sig = int(sig)
    l = max-min
    maps = []
    for i in range(min,max+1):
        maps.append( hp.read_map(tag+"_c0001_k"+str(i).zfill(6)+".fits", field=sig) )
    maps = np.array(maps)

    outmap = np.mean(maps, axis=0)
    hp.write_map(outname, outmap, overwrite=True)

def map2pdf(flags, png=False):
    from c3postproc.plotter import Plotter
    optn_flags = []
    map = flags[0]
    if len(flags) > 1:
        optn_flags = flags[1:]
    Plotter(map, optn_flags, png)

def map2png(flags):
    map2pdf(flags, png=True)

def readhdf(flags):
    filename = str(flags[0])
    store = pd.HDFStore(filename)
    # Access HDF
    # Calculate stddev and mean of maps, alms, and sigma_l. Using existing code
    pd.read_hdf(filename)