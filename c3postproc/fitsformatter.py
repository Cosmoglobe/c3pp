import time
import sys
import numpy as nu_p_map
import healpy as hp
import fitsio
from c3postproc.tools import *

def format_fits(chain, extname, types, units, nside, burnin, polar, component, fwhm, nu_ref, ProcVer, filename,):
    print(f"Formatting and outputting {filename}")
    h_dict = get_header(extname, types, units, nside, polar, component, fwhm, nu_ref, ProcVer, filename)
    dset = get_data(chain, extname, burnin, fwhm, nside)    

    fits = fitsio.FITS(f"{procver}/{filename}", mode="rw", clobber=True, verbose=True)
    fits.write(dset, header=h_dict, clobber=True)

def get_data(chain, extname, burnin, fwhm, nside):
    if extname.endswith('CMB'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="cmb/amp_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="cmb/amp_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)

        dset = np.asarray([amp_mean, beta_mean, amp_stddev, beta_stddev], dtype="f4")

    elif extname.endswith('SYNCHROTRON'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="synch/amp_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)
        beta_mean = h5handler(input=chain, dataset="synch/beta_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="synch/amp_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)
        beta_stddev = h5handler(input=chain, dataset="synch/beta_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)

        dset = np.asarray([amp_mean, beta_mean, amp_stddev, beta_stddev], dtype="f4")
         
    elif extname.endswith('FREE-FREE'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="ff/amp_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)
        EM_mean = h5handler(input=chain, dataset="ff/EM_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)
        Te_mean = h5handler(input=chain, dataset="ff/Te_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="ff/amp_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)
        EM_stddev = h5handler(input=chain, dataset="ff/EM_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)
        Te_stddev = h5handler(input=chain, dataset="ff/Te_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)

        dset = np.asarray([amp_mean, EM_mean, Te_mean, amp_stddev, EM_stddev, Te_stddev], dtype="f4")

    elif extname.endswith('AME'):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="ame/amp_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)
        beta_mean = h5handler(input=chain, dataset="ame/nu_p_map", min=burnin, max=None, smooth=fwhm, output=None, nside=nside, command=np.mean)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="ame/amp_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)
        nu_p_stddev = h5handler(input=chain, dataset="synch/nu_p_map", min=burnin, max=None, smooth=fwhm, output=None,nside= nside,command= np.std)

        dset = np.asarray([amp_mean, nu_p_mean, amp_stddev, nu_p_stddev], dtype="f4")


    return dset

def get_header(extname, types, units, nside, polar, component, fwhm, nu_ref, ProcVer, filename):

    stamp = f'Written {time.strftime("%c")}'
    
    h_dict = []

    h_dict.append({"name": "COMMENT"})
    h_dict.append({"name": "COMMENT", "value": "*** END of mandatory fields ***"})
    h_dict.append({"name": "COMMENT"})
                                                  
    h_dict.append({"name": "EXTNAME", "value": extname, "comment": "Extension name"})
    h_dict.append({"name": "DATE", "value": stamp, "comment": "Creation date"})
     
    h_dict.append({"name": "COMMENT"})
    h_dict.append({"name": "COMMENT", "value": "*** Column names ***"})
    h_dict.append({"name": "COMMENT"})
                                                  
    for i, type in enumerate(types):
        temp_dict = {}
        temp_dict["name"] = f"TTYPE{i}"
        temp_dict["value"] = type
        temp_dict["comment"] = type # TODO format comment
        h_dict.append(temp_dict)

    h_dict.append({"name": "COMMENT"})
    h_dict.append({"name": "COMMENT", "value": "*** Column units ***"})
    h_dict.append({"name": "COMMENT"})
                                                  
    for i, unit in enumerate(units):
        temp_dict = {}
        temp_dict["name"] = f"TUNIT{i}"
        temp_dict["value"] = unit
        temp_dict["comment"] = unit
        h_dict.append(temp_dict)

    h_dict.append({"name": "COMMENT"})
    h_dict.append({"name": "COMMENT", "value": "*** Planck params ***"})
    h_dict.append({"name": "COMMENT"})

    h_dict.append({"name": "PIXTYPE", "value": "HEALPIX", "comment": "HEALPIX pixelation"})
    h_dict.append({"name": "COORDSYS", "value": "GALACTIC", "comment": "Galactic coordinates"})
    h_dict.append({"name": "POLAR", "value": polar, "comment": "Polarization included"})
    h_dict.append({"name": "BAD_DATA", "value": hp.UNSEEN, "comment": "Sentinel value given to bad pixel."})
    h_dict.append({"name": "METHOD", "value": "COMMANDER", "comment": "Component separation method."})
    h_dict.append({"name": "AST-COMP", "value": component, "comment": "Astrophysical component name."})
    h_dict.append({"name": "FWHM", "value": fwhm, "comment": "FWHM used, arcmin."})
    h_dict.append({"name": "NU_REF", "value": nu_ref, "comment": "Reference frequency."})
    h_dict.append({"name": "PROCVER", "value": ProcVer, "comment": "Product version."})
    h_dict.append({"name": "FILENAME", "value": filename, "comment": "FITS filename."})

    return h_dict