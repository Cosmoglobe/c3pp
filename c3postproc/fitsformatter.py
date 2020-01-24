import time
import sys
import os
import numpy as np
import healpy as hp
import fitsio
from c3postproc.tools import *

def format_fits(chain, extname, types, units, nside, burnin, polar, component, fwhm, nu_ref, procver, filename,):
    print()
    print(f"Formatting and outputting {filename}")
    header = get_header(extname, types, units, nside, polar, component, fwhm, nu_ref, procver, filename)
    dset = get_data(chain, extname, burnin, fwhm, nside, types)    

    print(f"{procver}/{filename}", dset.shape)
    hp.write_map(f"{procver}/{filename}",dset,column_names=types, column_units=units, coord='G', overwrite=True, extra_header=header)

def get_data(chain, extname, burnin, fwhm, nside, types):
    if extname.endswith('CMB'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="cmb/amp_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="cmb/amp_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)

        # Masks
        mask1 = np.zeros((hp.nside2npix(nside)))
        mask2 = np.zeros((hp.nside2npix(nside)))

        dset = np.zeros((len(types),hp.nside2npix(nside)))

        dset[0] = amp_mean[0,:]
        dset[1] = amp_mean[1,:]
        dset[2] = amp_mean[2,:]
        
        dset[3] = amp_stddev[0,:]
        dset[4] = amp_stddev[1,:]
        dset[5] = amp_stddev[2,:]

        dset[6] = mask1
        dset[7] = mask2

    elif extname.endswith('SYNCHROTRON'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="synch/amp_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)
        beta_mean = h5handler(input=chain, dataset="synch/beta_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="synch/amp_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)
        beta_stddev = h5handler(input=chain, dataset="synch/beta_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)

        dset = np.zeros((len(types),hp.nside2npix(nside)))

        dset[0] = amp_mean[0,:]
        dset[1] = amp_mean[1,:]
        dset[2] = amp_mean[2,:]
        
        dset[3] = beta_mean[0,:]
        dset[4] = beta_mean[1,:]
        
        dset[5] = amp_stddev[0,:]
        dset[6] = amp_stddev[1,:]
        dset[7] = amp_stddev[2,:]
        
        dset[8]  = beta_stddev[0,:]
        dset[9]  = beta_stddev[1,:]
         
    elif extname.endswith('FREE-FREE'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="ff/amp_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)
        Te_mean = h5handler(input=chain, dataset="ff/Te_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="ff/amp_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)
        Te_stddev = h5handler(input=chain, dataset="ff/Te_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)

        dset = np.zeros((len(types),hp.nside2npix(nside)))

        dset[0] = amp_mean[0]
        dset[1] = Te_mean[0]
        
        dset[2] = amp_stddev[0]
        dset[3] = Te_stddev[0]
        
    elif extname.endswith('AME'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="ame/amp_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)
        nu_p_mean = h5handler(input=chain, dataset="ame/nu_p_map", min=burnin, max=None, output="map",fwhm=fwhm,  nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="ame/amp_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)
        nu_p_stddev = h5handler(input=chain, dataset="ame/nu_p_map", min=burnin, max=None, output="map",fwhm=fwhm, nside= nside,command= np.std)
        
        dset = np.zeros((len(types),hp.nside2npix(nside)))

        dset[0] = amp_mean[0]
        dset[1] = nu_p_mean[0]
        
        dset[2] = amp_stddev[0]
        dset[3] = nu_p_stddev[0]

    print(f"Shape of dset {dset.shape}")
    return dset

def get_header(extname, types, units, nside, polar, component, fwhm, nu_ref, procver, filename):

    stamp = f'Written {time.strftime("%c")}'
    
    header = []
    header.append(("PIXTYPE", "HEALPIX"))
    header.append(("COORDSYS", "GALACTIC"))
    header.append(("POLAR", polar))
    header.append(("BAD_DATA", hp.UNSEEN))
    header.append(("METHOD", "COMMANDER"))
    header.append(("AST-COMP", component))
    header.append(("FWHM", fwhm))
    header.append(("NU_REF", nu_ref))
    header.append(("PROCVER", procver))
    header.append(("FILENAME", filename))

    return header