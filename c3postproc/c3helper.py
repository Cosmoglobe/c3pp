import math
import sys
import os
import time
import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def c3help(op, nsysargs):
    if str(op) == "stddev":
        if nsysargs < 7:
            print("Usage: ")
            print("c3postproc stddev [input tag: cmb, dust, dust_t etc.] [signal T/Q/U or 1/2/3] [sample range min] [sample range max] [outname]")
            print("This thing calculates the standard deviation for all sky maps.")
            sys.exit()
    elif str(op) == "mean":
        if nsysargs < 7:
            print("Usage: ")
            print("c3postproc mean [input tag: cmb, dust, dust_t etc.] [signal T/Q/U or 1/2/3] [sample range min] [sample range max] [outname]")
            print("This thing calculates the mean for all sky maps.")
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
    elif str(op) == "readhdf":
        pass
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
