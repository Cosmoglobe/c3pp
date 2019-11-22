import sys
import os

def c3help(op, nsysargs):
    if str(op) == "stddev":
        if nsysargs < 7:
            print("Usage: ")
            print("c3pp stddev [.h5 filename] [h5 dataset (ex. cmb/sigma_l)] [sample range min] [sample range max] [output name (name.fits/name.dat)]")
            print("This thing calculates the standard deviation for any .h5 dataset quantity over a given sample range.")
            sys.exit()
    elif str(op) == "mean":
        if nsysargs < 7:
            print("Usage: ")
            print("c3pp mean [.h5 filename] [h5 dataset (ex. cmb/sigma_l)] [sample range min] [sample range max] [output name (name.fits/name.dat)]")
            print("This thing calculates the mean for any .h5 dataset quantity over a given sample range.")
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
            print("-min xx            ")
            print("-max xx            ")
            print("-range xx          ")
            print("-bar          adds colorbar (\"cb\" in filename)" )
            print("-lmin xx , set to 0 by default")
            print("-lmax xx , This is automatically set from the h5 file.")
            print("-fwhm xx ")
            print("-mask FILENAME         ")
            print("-I/-Q/-U/-QU/-IQU (-I default)         ")
            print("-logscale          ")
            print("-small -medium -large (8.8, 12.0, 18. All by default)          ")
            print("-white_background          ")
            print("-darkmode          Plots all outlines in white for dark bakgrounds (\"dark\" in filename)")
            print("-png , pdf is default")
            print()
            print("Plots map as png, autoregognizes type, if not autosets parameters. ")
            print("Most amplitudes will be plotted with a semi-logarithmic scale by default. You will be notified. ")
            print()
            sys.exit()            
    elif str(op) == "sigma_l2fits":
        if nsysargs < 5:
            print("Usage:   ")
            print("c3pp sigma_l2fits [.h5 filename] [h5 dataset (optional, default cmb/sigma_l)] [burn-in (sample at which finishes burn-in)] [out file (name.fits)]")
            print("This tool converts the .h5 dataset from Commander3 into .fits dateset suitable for Commander1 BR and GBR estimator analysis (See comm_like_tools for further information about BR and GBR post processing).")
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
    operations_list = ["stddev", "mean", "plot", "sigma_l2fits"]
    return operations_list
