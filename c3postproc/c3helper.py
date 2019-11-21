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
    elif str(op) == "map2pdf":
        if nsysargs < 3:
            print()
            print("Usage: ")
            print("c3pp map2pdf [input file]")
            print("optional: [-min xx] [-max xx] [-range xx] [-bar] [-mask FILENAME] [-I/-Q/-U/-QU/-IQU (-I default)] [-logscale] [-small -medium -large (8.8, 12.0, 18. All by default)]")
            print("Plots map as pdf, autoregognizes type, if not autosets parameters. ")
            print("Most amplitudes will be plotted with a semi-logarithmic scale by default. You will be notified. ")
            print()
            sys.exit()
    elif str(op) == "map2png":
        if nsysargs < 3:
            print()
            print("Usage:  ")
            print("c3pp map2png [(inputfile.fits) / inputfile.h5 h5-group-path nside] ")
            print("example:  ")
            print("c3pp map2png chain_c0001.h5 000004/cmb/amp_alm 2048 -bar -darkmode -medium -IQU")
            print("c3pp map2png cmb_c0001_k000004.fits -white_background")
            print()
            print("optional:        " )
            print("-min xx            ")
            print("-max xx            ")
            print("-range xx          ")
            print("-bar           ")
            print("-lmax xx ")
            print("-fwhm xx ")
            print("-mask FILENAME         ")
            print("-I/-Q/-U/-QU/-IQU (-I default)         ")
            print("-logscale          ")
            print("-small -medium -large (8.8, 12.0, 18. All by default)          ")
            print("-white_background          ")
            print("-darkmode")
            print()
            print("Plots map as png, autoregognizes type, if not autosets parameters. ")
            print("Most amplitudes will be plotted with a semi-logarithmic scale by default. You will be notified. ")
            print()
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
    operations_list = ["stddev", "mean", "map2pdf", "map2png"]
    return operations_list
