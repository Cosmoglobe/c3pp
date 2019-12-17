# Commander3 postprocessing tool (c3pp)

A python code for processing commander3 files

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
c3pp
````
in your terminal. This will display a list of all available tools.
For more information on each specific tool, simply type
```bash
c3pp [toolname]
````
and a description will be printed.

### Currently available tools
#### Mean and stddev
Both these tools take a .h5-file and calculate either mean or standard deviation of a signal (such as CMB or dust, or sigma_l) over a range of samples specified by the command line arguments. Depending on wether the input signal is a map or a list of numbers, the 
data will be stored in a .fits or a .dat file. Map values can also be smoothed using the -smooth function. For stddev, these are smoothed per sample before calculation, for mean, the mean map is smoothed.

#### plot
This feature allows for plotting any .h5 map signal to a pdf of png file with a template similar to that of the Planck collaboration. Many features are availble here, such as plotting with white outlines, transparent background, . Additionally, when applying this to commander data, the input name is automatically identified and range, color and scale is automatically set to the standard plotting format, which means you may simply specify the filename and any optional arguments such as which sky signal, size of plot etc..

Furthermore, plotting a map directly from alms will also soon be available, which allows for direct smoothing and optional nside.

#### alm2fits
Exports alms from .h5 to a separate fits file.

#### sigma_l2fits
Exports sigma_ls from .h5 to a separate .fits file.

#### dlbin2dat
Takes Dl signals from an .h5 file and calculates mean and standard deviation per bin given a binfile and outputs to a .dat.

#### h5map2fits
Exports a .h5 map signal to a separate .fits file.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)