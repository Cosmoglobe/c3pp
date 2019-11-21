import sys
import os

def c3help(op, nsysargs):
    if str(op) == "stddev":
        if nsysargs < 7:
            print("Usage: ")
            print("c3postproc stddev [.h5 filename] [h5 dataset (ex. cmb/sigma_l)] [sample range min] [sample range max] [output name (name.fits/name.dat)]")
            print("This thing calculates the standard deviation for any .h5 dataset quantity over a given sample range.")
            sys.exit()
    elif str(op) == "mean":
        if nsysargs < 7:
            print("Usage: ")
            print("c3postproc mean [.h5 filename] [h5 dataset (ex. cmb/sigma_l)] [sample range min] [sample range max] [output name (name.fits/name.dat)]")
            print("This thing calculates the mean for any .h5 dataset quantity over a given sample range.")
            sys.exit()
    elif str(op) == "map2pdf":
        if nsysargs < 3:
            print()
            print("Usage: ")
            print("c3postproc map2pdf [input file]")
            print("optional: [-min xx] [-max xx] [-range xx] [-bar] [-mask FILENAME] [-I/-Q/-U/-QU/-IQU (-I default)] [-logscale] [-small -medium -large (8.8, 12.0, 18. All by default)]")
            print("Plots map as pdf, autoregognizes type, if not autosets parameters. ")
            print("Most amplitudes will be plotted with a semi-logarithmic scale by default. You will be notified. ")
            print()
            sys.exit()
    elif str(op) == "map2png":
        if nsysargs < 3:
            print()
            print("Usage: ")
            print("c3postproc map2png [input file]")
            print("optional: [-min xx] [-max xx] [-range xx] [-bar] [-mask FILENAME] [-I/-Q/-U/-QU/-IQU (-I default)] [-logscale] [-small -medium -large (8.8, 12.0, 18. All by default)] [-white_background] [-darkmode]")
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
