This repository contains the Python-based code to calculate synthetic seismograms for simulated volcanic 
eruptions. The code carries out steps 2 and 3 in the workflow presented in Coppess et al (2022). 
Pressure and wall shear traction changes for a conduit-chamber system are converted to moment tensor and 
vertical force histories, using the representation theorem developed in Coppess et al (2022). The source
histories are then convolved with numerical elastic Green's functions to calculate seismograms for 
specified receiver positions.

## Download and install
To install the package, go to the desired directory and use the following command: 
```
git clone git@bitbucket.org:kcoppess/synthetic-seismograms.git
```
Once all of the correct dependencies are installed (see below), the code should be ready to use 
immediately. ***Be sure to only use conduit-flow branch.*** Other branches will not have the same
structure.

### Dependencies
This Python package was developed with the following dependencies: python-3.9.12, numpy-1.21.5, 
matplotlib-3.5.1, scipy-1.8.1, argparse-1.1

## Running the code
There are two ways to run: (1) loading in arguments using an input file or (2) using command-line 
directly. Using the command `python main.py -h` gives usage information for both options. You should 
see something like the following printed:
```
usage: main.py [-h] [-s SAVE] [-p] [--total_time TOTAL_TIME] [--dt DT] [--MTGF MTGF] [--SFGF SFGF]
               [--sourcedepth SOURCEDEPTH] [--chamvol CHAMVOL] [--condrad CONDRAD] [--stations STATIONS]
               sim path stp rep con der

Calculating synthetic seismograms. Two options for inputting parameters: (1) load in from file: main.py
@[argument-file].txt (see ex_args.txt for example) or (2) see usage above.

positional arguments:
  sim                   simulation label (used as path_to_simulation/<sim>.zip)
  path                  path to directory where simulation results stored
  stp                   source type (options: CHAMBER or CONDUIT)
  rep                   PS (point source representation) or ES (extended source representation; only used
                        for conduit)
  con                   calculate force and/or moment contributions to seismogram (options: FORCE, MOMENT,
                        BOTH)
  der                   ACC (returns acceleration seismograms), VEL (velocity), DIS (displacement)

optional arguments:
  -h, --help            show this help message and exit
  -s SAVE, --save SAVE  path to directory where synthetic seismograms and force/moment histories are saved
                        (default: no saving)
  -p, --plot            display plot of synthetic seismograms (default: False)
  --total_time TOTAL_TIME
                        total time in seconds for synthetic seismograms (default: 1500)
  --dt DT               time step size in seconds (needs to be >= GF time step size) (default: 0.04)
  --MTGF MTGF           path to directory storing moment tensor Greens functions (default: see main.py)
  --SFGF SFGF           path to directory storing single force Greens functions (default: see main.py)
  --sourcedepth SOURCEDEPTH
                        depth of point source in meters (assumes on z-axis; only relevant for frankenstein
                        analytical calculation): 500m (conduit) or 1028.794m (chamber) (default: -173)
  --chamvol CHAMVOL     magma chamber volume in m^3 (default: 100000.0)
  --condrad CONDRAD     cylindrical conduit radius in meters (default: 30)
  --stations STATIONS   file with station labels and coordinates with origin at center of conduit vent
                        (default: station_pos.txt)
```
The positional arguments are required to run, while the optional arguments will be set to the default
values specified. Medium parameters are set directly in `main.py` (see starting at line 155).

### Required input data
Example simulation data files can be found on [Open Science Framework](https://doi.org/10.17605/OSF.IO/R6HJC).
Currently, input files are assumed to be in .zip format (functionality for .mat formats is in the works). 

Required data files:

    - `height.txt` : depths of grid points tracked in simulation (assumes uniform spacing and positive is up and bottom of conduit is at 0)
    - `time.txt` : time points
    - for MOMENT calculations:
        - CONDUIT: `pressure.txt` : array of pressure along conduit with shape (# of grid points, # of time points)
        - CHAMBER: `chamber_pressure.txt` : chamber pressure at each time point with shape (# of time points)
    - for FORCE calculations:
        - CONDUIT: `wall_trac.txt` : array of shear traction acting ON magma (i.e. opposite sign of on earth) with same shape as pressure
        - CHAMBER:
            - `chamber_pressure.txt` : see above
            - `density.txt` : magma density (same shape as pressure)
            - `velocity.txt` : magma particle velocity (same shape as pressure)

Loaded data will then be processed to smooth out any numerical effects (e.g. from downsampling), as well
as interpolate to get constant time-stepping (which is assumed in the remaining workflow). See 
`load_data.py` for how data is loaded in and processed. 

## Green's functions
Default Green's functions were calculated using code base developed by Zhu, L., & Rivera, L. A. (2002).
These are stored in the directory `greens_functions/halfspace`. The medium and other input parameters
used are noted in `specs.txt`, along with source and receiver positions that have been calculated.
Point-source GF are labeled as `halfA_{source depth}_{mt|sf}` with `mt` noting moment tensor GF and `sf`
noting single force GF. Extended-source GF are labeled as `extended_{source extent}_{mt|sf}`. Within
each of these directories are directories labeled by receiver position relative to vent (can also be
labeled by station name).

New Green's functions can be stored in `greens_functions` directory. If the source positions used to 
generate the GFs are not the same as the grid positions in the simulation dataset, there is an option
to interpolate the loaded GFs to get the values at the desired grid positions (i.e. values stored in 
`height.txt`). This is turned on by default (see lines 201, 205, 222 and 226 in `main.py`). For more 
specifics, see function call description in `load_gfs.py`.

