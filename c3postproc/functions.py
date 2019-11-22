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

def h52fits(flags):
    filename = str(flags[0])
    path     = "cmb/sigma_l"
    burnin  = int(flags[-2])
    outname  = str(flags[-1])
    if len(flags)==4:
        path = str(flags[1])

    import h5py
    with h5py.File(filename, 'r') as f:
        print("Reading HDF5 file: "+filename+" ...")
        groups = list(f.keys())
        print()
        print("Reading "+str(len(groups))+" samples from file.")

        dset = np.zeros((len(groups)+1,1,len(f[groups[0]+'/'+path]),len(f[groups[0]+'/'+path][0])))
        nspec = len(f[groups[0]+'/'+path])
        lmax  = len(f[groups[0]+'/'+path][0])-1

        print('Found: \
        \npath in the HDF5 file : '+path+' \
        \nnumber of spectra :'+str(nspec)+\
        '\nlmax: '+str(lmax) )

        for i in range(len(groups)):
            for j in range(nspec):
                dset[i+1,0,j,:] = np.asarray(f[groups[i]+'/'+path][j][:])

    ell = np.arange(lmax+1)
    for i in range(1,len(groups)+1):
        for j in range(nspec):
            dset[i,0,j,:] = dset[i,0,j,:]*ell[:]*(ell[:]+1.)/2./np.pi
    dset[0,:,:,:] = len(groups) - burnin

    import fitsio
    print("Dumping fits file: "+outname+" ...")
    dset = np.asarray(dset, dtype='f4')
    fits = fitsio.FITS(outname,mode='rw',clobber=True, verbose=True)
    h_dict = [{'name':'FUNCNAME','value':'Gibbs sampled power spectra','comment':'Full function name'}, \
              {'name':'LMAX','value':lmax,'comment':'Maximum multipole moment'}, \
              {'name':'NUMSAMP','value':len(groups),'comment':'Number of samples'}, \
              {'name':'NUMCHAIN','value':1,'comment':'Number of independent chains'}, \
              {'name':'NUMSPEC','value':nspec,'comment':'Number of power spectra'}]
    fits.write(dset[:,:,:,:],header=h_dict,clobber=True)
    fits.close()
    
