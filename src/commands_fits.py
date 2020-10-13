import time
import os
import numpy as np
import sys
import click
from src.tools import *

@click.group()
def commands_fits():
    pass

@commands_fits.command()
@click.argument("input", type=click.STRING)
def printheader(input,):
    """
    Prints the header of a fits file.
    """
    from astropy.io import fits

    with fits.open(input) as hdulist:
        hdulist.info()
        for hdu in hdulist:
            print(repr(hdu.header))

@commands_fits.command()
@click.argument("input", type=click.STRING)
def printdata(input,):
    """
    Prints the data of a fits file
    """
    from astropy.io import fits

    with fits.open(input) as hdulist:
        hdulist.info()
        for hdu in hdulist:
            print(repr(hdu.data))


@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.argument("columns", type=click.INT)
def rmcolumn(input, output, columns):
    """
    remove columns in fits file
    """
    from astropy.io import fits

    with fits.open(input) as hdulist:
        for hdu in hdulist:
            hdu.header.pop(columns)        
            hdu.data.del_col(columns)        
        hdulist.writeto(output, overwrite=True)
        

@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
def QU2ang(input, output,):
    """
    remove columns in fits file
    """
    import healpy as hp
    Q, U = hp.read_map(input, field=(1,2), dtype=None, verbose=False)
    phi = 0.5*np.arctan(U,Q)
    hp.write_map(output, phi, dtype=None, overwrite=True)





@commands_fits.command()
@click.argument("input1", type=click.STRING)
@click.argument("input2", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-beam1", type=click.STRING, help="Optional beam file for input 1",)
@click.option("-beam2", type=click.STRING, help="Optional beam file for input 2",)
@click.option("-mask", type=click.STRING, help="Mask",)
def crosspec(input1, input2, output, beam1, beam2, mask,):
    """
    This function calculates a powerspectrum from polspice using this path:
    /mn/stornext/u3/trygvels/PolSpice_v03-03-02/
    """
    sys.path.append("/mn/stornext/u3/trygvels/PolSpice_v03-03-02/")
    from ispice import ispice

    lmax = 6000
    fwhm = 0
    #mask = "dx12_v3_common_mask_pol_005a_2048_v2.fits"
    if beam1 and beam2:
        ispice(input1,
               clout=output,
               nlmax=lmax,
               beam_file1=beam1,
               beam_file2=beam2,
               mapfile2=input2,
               maskfile1=mask,
               maskfile2=mask,
               fits_out="NO",
               polarization="YES",
               subav="YES",
               subdipole="YES",
               symmetric_cl="YES",
           )
    else:

        ispice(input1,
               clout=output,
               nlmax=lmax,
               beam1=0.0,
               beam2=0.0,
               mapfile2=input2,
               maskfile1=mask,
               maskfile2=mask,
               fits_out="NO",
               polarization="YES",
               subav="YES",
               subdipole="YES",
               symmetric_cl="YES",
           )

@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-minchain", default=1, help="lowest chain number, c0002 [ex. 2] (default=1)",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5] (default=1)",)
@click.option("-chaindir", default=None, type=click.STRING, help="Base of chain directory, overwrites chain iteration from input file name to iteration over chain directories, BP_chain_c15 to BP_chain_c19 [ex. 'BP_chain', with minchain = 15 and maxchain = 19]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for down-grading maps before calculation",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-missing", is_flag=True, help="If files are missing, drop them. Else, exit computation",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def fits_mean(
        input, output, min, max, minchain, maxchain, chaindir, fwhm, nside, zerospin, missing, pixweight):
    """
    Calculates the mean over sample range from fits-files.\n
    ex. res_030_c0001_k000001.fits res_030_20-100_mean_40arcmin.fits -min 20 -max 100 -fwhm 40 -maxchain 3\n
    If output name is set to .dat, data will not be converted to map.

    Note: the input file name must have the 'c0001' chain identifier and the 'k000001' sample identifier. The -min/-max and -chainmin/-chainmax options set the actual samples/chains to be used in the calculation 
    """

    fits_handler(input, min, max, minchain, maxchain, chaindir, output, fwhm, nside, zerospin, missing, pixweight, False, np.mean)

@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-minchain", default=1, help="lowest chain number, c0002 [ex. 2] (default=1)",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5] (default=1)",)
@click.option("-chaindir", default=None,type=click.STRING, help="Base of chain directory, overwrites chain iteration from input file name to iteration over chain directories, BP_chain_c15 to BP_chain_c19 [ex. 'BP_chain', with minchain = 15 and maxchain = 19]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for down-grading maps before calculation",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-missing", is_flag=True, help="If files are missing, drop them. Else, exit computation",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def fits_stddev(
        input, output, min, max, minchain, maxchain, chaindir, fwhm, nside, zerospin, missing, pixweight):
    """
    Calculates the standard deviation over sample range from fits-files.\n
    ex. res_030_c0001_k000001.fits res_030_20-100_mean_40arcmin.fits -min 20 -max 100 -fwhm 40 -maxchain 3\n
    If output name is set to .dat, data will not be converted to map.

    Note: the input file name must have the 'c0001' chain identifier and the 'k000001' sample identifier. The -min/-max and -chainmin/-chainmax options set the actual samples/chains to be used in the calculation 
    """

    fits_handler(input, min, max, minchain, maxchain, chaindir, output, fwhm, nside, zerospin, missing, pixweight, False, np.std)
