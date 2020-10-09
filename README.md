# Commander3 postprocessing tool (c3pp)

A python code for processing and plotting _commander3_ files.
![Masterpiece](https://github.com/trygvels/c3pp/blob/master/imgs/spectrum.pdf  =250x)
![Masterpiece](https://github.com/trygvels/c3pp/blob/master/imgs/spectrum_pol.pdf  =250x)


## Installation

Install tools by running 

```bash
pip install git+https://github.com/trygvels/c3pp.git -U
```

or 


```bash
pip install git+https://github.com/trygvels/c3pp.git --user
```

and make sure that
```bash
PATH=$PATH:~/.local/bin
````
is in your path.


## Usage
The tool uses command line arguments and has many different tools built in (and more coming).
In order to get an overview of the available tools and how to access them, simply type
```bash
c3pp --help
````
in your terminal. This will display a list of all available tools.
For more information on each specific tool, simply type
```bash
c3pp [toolname] --help
````
and a description will be printed:
```
[command prompt]$ c3pp --help
Usage: c3pp [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  alm2fits          Converts c3 alms in .h5 file to fits with given nside and optional...
  crosspec          This function calculates a powerspectrum from polspice using this path:...
  dlbin2dat         Outputs a .dat file of binned powerspectra averaged over a range of output...
  fits-mean         Calculates the mean over sample range from fits-files.
  fits-stddev       Calculates the standard deviation over sample range from fits-files.
  generate-sky      Generate sky maps from separate input maps.
  gnomplot
  h52fits           Outputs a .h5 map to fits on the form 000001_cmb_amp_n1024.fits
  make-diff-plots   Produces standard c3pp plots from the differences between two output...
  mean              Calculates the mean over sample range from .h5 file.
  output-sky-model  Outputs spectrum plots c3pp output-sky-model -a_s synch_c0001_k000100.fits...
  pixreg2trace      Outputs the values of the pixel regions for each sample to a dat file.
  plot              Plots map from .fits or h5 file.
  plotrelease       Plots all release files
  printdata         Prints the data of a fits file
  printheader       Prints the header of a fits file.
  qu2ang            remove columns in fits file
  release           Creates a release file-set on the BeyondPlanck format.
  rmcolumn          remove columns in fits file
  sigma-l2fits      Converts c3-h5 dataset to fits suitable for c1 BR and GBR estimator...
  specplot          This function plots the file output by the Crosspec function.
  stddev            Calculates the stddev over sample range from .h5 file.
  traceplot         This function plots a traceplot of samples from min to max with optional...
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)