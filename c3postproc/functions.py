import math
import sys
import os
import time
import healpy as hp
import numpy as np

def h5handler(flags, command):
    filename = str(flags[0])
    signal = str(flags[1])
    min = int(flags[2])
    max = int(flags[3])
    outname = flags[-1]
    l = max-min
    
    import h5py     
    dats = []
    with h5py.File(filename, 'r') as f:
        for i in range(min,max+1):
            s = str(i).zfill(6)
            dats.append(f[s+"/"+signal][()])
    dats = np.array(dats)

    outdata = command(dats, axis=0)
    if "fits" in outname[-4:]: 
        hp.write_map(outname, outdata, overwrite=True)
    else:
        np.savetxt(outname, outdata)

def mean(flags):
    h5handler(flags, np.mean)

def stddev(flags):
    h5handler(flags, np.std)


def plot(flags):
    from c3postproc.plotter import Plotter
    Plotter(flags)


        