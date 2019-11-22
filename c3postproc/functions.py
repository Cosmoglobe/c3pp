import math
import sys
import os
import time
import healpy as hp
import numpy as np



#######################
# ACTUAL MODULES HERE #
#######################



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

def h52fits(flags, save=True):
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

    if save:
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
    return dset

def h5map2fits(flags, save=True):
    import h5py
    h5file = str(flags[0])
    dataset = get_key(flags, h5file) 
    with h5py.File(h5file, 'r') as f:
        maps = f[dataset][()]
        lmax = f[dataset[:-4]+"_lmax"][()] # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    outfile =  dataset.replace("/", "_")
    outfile = outfile.replace("_map","")
    if save:
        hp.write_map(outname, data)
    return maps, nside, lmax, outfile

def alm2fits(flags, save=True):
    import h5py
    h5file = str(flags[0])
    dataset = get_key(flags, h5file) 

    with h5py.File(h5file, 'r') as f:
        maps = f[dataset][()]
        lmax = f[dataset[:-4]+"_lmax"][()] # Get lmax from h5
    
    if "-lmax" in flags:
        lmax_ = int(get_key(flags, "-lmax"))
        if lmax_ > lmax:
            print("lmax larger than data allows: ", lmax)
            print("Please chose a value smaller than this")
        else:
            lmax =  lmax_
        print("Setting lmax to ", lmax)
        mmax = lmax
    else:
        mmax = lmax

    if "-fwhm" in flags:
        fwhm = float(get_key(flags, "-fwhm"))
    else:
        fwhm = 0.0


    alms = unpack_alms(data,lmax) # Unpack alms
    nside = int(get_key(flags, dataset)) # Output nside
    
    hehe = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1) 
    print("alm length hehe:",hehe)
    print("alm length actual:", alms.shape)

    data = hp.sphtfunc.alm2map(alms, nside, lmax=lmax, fwhm=arcmin2rad(fwhm))

    outfile =  dataset.replace("/", "_")
    outfile = outfile.replace("_alm","")
    if save:
        hp.write_map(outname, data)
    return data, nside, lmax, fwhm, outfile



#######################
# HELPFUL TOOLS BELOW #
#######################



def unpack_alms(maps,lmax):
    mind = []
    lm = []
    idx = 0

    for m in range(0,lmax):
        mind.append(idx)
        if m == 0:
            for l in range(m, lmax):
                #lm[idx] = (l,m)
                lm.append((l,m))
                idx += 1
        else:
            for l in range(m,lmax):
                #lm(idx) = (l,m)
                lm.append((l,m))
                idx +=1
                
                #lm(idx) = (l,-m)
                lm.append((l,-m))
                idx +=1
    print(len(lm))

    alms =[] 
    pol = 0
    #for l in range(lmax):
    #    for m in range(-l,l):
    

    # 151 leftover when lmax=150! WHAT DOES THAT MEAN?!

    for i in range(len(lm)):
        l = lm[i][0]
        m = lm[i][1]
        if m < 0:
            continue
        idx = lm2i(l,m,mind)

        if m == 0:
            alms.append(complex( maps[pol, idx], 0.0 ))
        else:  
            #alms.append( 1/np.sqrt(2)*complex(maps[pol,idx], maps[pol, lm2i(l,-m,mind) ]) )
            alms.append( complex(maps[pol,idx], maps[pol, idx+1])/np.sqrt(2) )
    print(len(alms))
    return np.array(alms, dtype=np.complex128)

def lm2i(l,m, mind):
    if l>150 or abs(m)> l:
        print("HELLO1")
        sys.exit()
    if  mind[abs(m)]==-1:
        print("HELLO2")
        sys.exit()
    if abs(m) > l:
        print("HELLO3")
        sys.exit()

    if m == 0:
        i = mind[m] + l 
    else: 
        i = mind[abs(m)] + 2*(l-abs(m))
        if m<0:
            i += 1 
    return i

def get_key(flags, keyword):
    return flags[flags.index(keyword) + 1]