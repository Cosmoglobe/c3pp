import sys
import os

def c3help(op, nsysargs):
    if str(op) == "stddev":
        if nsysargs < 7:
            print()
            print("Usage: ")
            print("c3pp stddev [.h5 filename] [h5 dataset (ex. cmb/sigma_l)] [sample range min] [sample range max] (OPTIONAL: [-smooth FWHM]) [output name (name.fits/name.dat)]")
            print("example: ")
            print("c3pp stddev chain_c0001.h5 cmb/amp_map 10 20 -smooth 60 cmb_amp_60arcmin.fits")
            print("This thing calculates the standard deviation for any .h5 dataset quantity over a given sample range.")
            print()
            sys.exit()
    elif str(op) == "mean":
        if nsysargs < 7:
            print()
            print("Usage: ")
            print("c3pp mean [.h5 filename] [h5 dataset (ex. cmb/sigma_l)] [sample range min] [sample range max] [output name (name.fits/name.dat)]")
            print("This thing calculates the mean for any .h5 dataset quantity over a given sample range.")
            print()
            sys.exit()
    elif str(op) == "plot":
        if nsysargs < 3:
            print()
            print("Usage:  ")
            print("c3pp plot [(inputfile.fits) / inputfile.h5 h5-group-path nside] ")
            print("example:  ")
            print("c3pp plot chain_c0001.h5 000004/cmb/amp_alm 2048 -bar -darkmode -medium -QU")
            print("c3pp plot cmb_c0001_k000004.fits -white_background")
            print()
            print("optional:        " )
            print("-min xx                  Min value of colorbar, overrides autodetector.")
            print("-max xx                  Max value of colorbar, overrides autodetector.")
            print("-range xx                Color range. \"-range auto\" sets to 95 percentile of data.")
            print("-bar                     Adds colorbar (\"cb\" in filename)" )
            print("-lmax xx ,               This is automatically set from the h5 file. Only available for alm inputs.")
            print("-fwhm xx                 FWHM of smoothing to apply to alm binning. Only available for alm inputs.")
            print("-mask FILENAME           Masks input with specified maskfile.")
            print("-mfill color             Color to fill masked area. for example \"gray\". Transparent by default.")
            print("-I/-Q/-U/-QU/-IQU        Signal to be plotted (-I default)         ")
            print("-logscale                Plots using planck semi-logscale. (Autodetector sometimes uses this.)")
            print("-small -medium -large    Size: 1/3, 1/2 and full page width. 8.8, 12.0, 18. cm (Medium by default)")
            print("-white_background        Sets the background to be white. (Transparent by default [recommended])")
            print("-darkmode                Plots all outlines in white for dark bakgrounds (\"dark\" in filename)")
            print("-pdf                     Saves output as .pdf ().png by default)")
            print()
            print("Plots map as png, autoregognizes type, if not autosets parameters. ")
            print("Most amplitudes will be plotted with a semi-logarithmic scale by default. You will be notified. ")
            print()
            sys.exit()            
    elif str(op) == "sigma_l2fits":
        if nsysargs < 6:
            print("Usage:   ")
            print("c3pp sigma_l2fits [.h5 file root (root_XXXXX.h5)] [h5 dataset (optional, default cmb/sigma_l)] [number of chains] [burn-in (sample at which finishes burn-in)] [out file (name.fits)]")
            print("This tool converts the .h5 dataset from Commander3 into .fits dateset suitable for Commander1 BR and GBR estimator analysis (See comm_like_tools for further information about BR and GBR post processing).")
            sys.exit()
    elif str(op) == "h5map2fits":
        if nsysargs < 3:
            print("Usage:   ")
            print("c3pp h5map2fits [.h5 filename] [h5 dataset (h5-group path)]")
            print("This tool outputs a h5 map to fits on the form 000001_cmb_amp_n1024.fits")
            sys.exit()
    elif str(op) == "alm2fits":
        if nsysargs < 4:
            print("Usage:   ")
            print("c3pp alm2fits [.h5 filename] [h5 dataset (h5-group path)] [nside] ")
            print("optional:        " )
            print("-lmax xx       chose a lower lmax if you wish  ")
            print("-fwhm xx       fwhm in arcmin for smoothing ")
            print("This tool converts the alms stored in the commander3 h5 file to fits, with given nside, lmax and fwhm.")
            sys.exit()
    elif str(op) == "help":
        c3help_general()
    else:
        print("Operation not recognized. Try", operations(), " or c3pp help")
        sys.exit()


def c3help_general():
    print("help:  c3pp [operation]")
    print("List of possible operations:")
    print(operations())
    sys.exit()

def operations():
    operations_list = ["stddev", "mean", "plot", "sigma_l2fits", "h5map2fits", "alm2fits"]
    return operations_list
