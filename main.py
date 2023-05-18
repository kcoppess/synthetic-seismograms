#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import argparse
from zipfile import ZipFile
import scipy.signal as sg
import scipy.interpolate as si

import load_data as ld
import point_source as PS
import extended_source as ES
import helpers as hp
import source_setup as ss
import source_plot as sp

'''
-------------------
COMMAND LINE INPUTS
-------------------
'''
parser = argparse.ArgumentParser(description='Calculating synthetic seismograms. Two options for inputting parameters: (1) load in from file: main.py @[argument-file].txt (see ex_args.txt for example) or (2) see usage above.',
                                 fromfile_prefix_chars='@',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
'''required inputs'''
parser.add_argument('sim', help='simulation label (used as path_to_simulation/<sim>.zip)')
parser.add_argument('path', help='path to directory where simulation results stored')
parser.add_argument('stp', help='source type (options: CHAMBER or CONDUIT)')
parser.add_argument('rep', help='PS (point source representation) or ES (extended source representation; only used for conduit)')
parser.add_argument('con', help='calculate force and/or moment contributions to seismogram (options: FORCE, MOMENT, BOTH)')
parser.add_argument('der', help='ACC (returns acceleration seismograms), VEL (velocity), DIS (displacement)')

'''optional inputs'''
parser.add_argument('-s', '--save', default='no saving',
                    help='path to directory where synthetic seismograms and force/moment histories are saved')
parser.add_argument('-p', '--plot', action='store_true', help='display plot of synthetic seismograms')
parser.add_argument('-total_time', default=730, type=float,
                    help='total time in seconds for synthetic seismograms')
parser.add_argument('-dt', default=0.04, type=float,
                    help='time step size in seconds (needs to be >= GF time step size)')
parser.add_argument('-MTGF', default='see main.py',
                    help='path to directory storing moment tensor Greens functions')
parser.add_argument('-SFGF', default='see main.py',
                    help='path to directory storing single force Greens functions')
parser.add_argument('-sourcedepth', default=-173, type=float, 
        help='depth of point source in meters (assumes on z-axis; only relevant for frankenstein analytical calculation): 500m (conduit) or 1028.794m (chamber)')
parser.add_argument('-chamvol', default=1e5, type=float, 
                    help='magma chamber volume in m^3')
parser.add_argument('-condrad', default=50, type=float,
                    help='cylindrical conduit radius in meters')
parser.add_argument('-stations', default='station_pos.txt', 
                    help='file with station labels and coordinates with origin at center of conduit vent')

args = parser.parse_args()

'''
---------------------------------------------------
Translating input arguments into relevant variables
---------------------------------------------------
'''

SIMULATION = args.sim
directory = args.path + SIMULATION
zip_filename = directory + '.zip'

SOURCE_TYPE = args.stp
REPRESENTATION = args.rep
CONTRIBUTION = args.con
DERIV = args.der

if args.save == 'no saving':
    SAVE = False
else:
    SAVE = True
    direc_base = args.save + SIMULATION
    if DERIV == 'DIS':
        direc = direc_base+'/displacement/'
    elif DERIV == 'VEL':
        direc = direc_base+'/velocity/'
    else:
        direc = direc_base+'/acceleration/'
    save_file = direc+SOURCE_TYPE+'__'+REPRESENTATION+'__'
    if not os.path.exists(direc):
        os.makedirs(direc)

PLOT = args.plot
TOTAL_TIME = args.total_time
DT = args.dt

if args.MTGF == 'see main.py':
    if SOURCE_TYPE == 'CHAMBER':
        MT_GF_FILE = '/Users/kcoppess/muspelheim/synthetic-seismograms/synthetic-seismograms/greens_functions/halfspace/halfA_chamber/halfA_1.028794_mt/'
    elif SOURCE_TYPE == 'CONDUIT':
        if REPRESENTATION == 'PS':
            MT_GF_FILE = '/Users/kcoppess/muspelheim/synthetic-seismograms/synthetic-seismograms/greens_functions/halfspace/halfA_conduit/halfA_0.50195_mt/'
        elif REPRESENTATION == 'ES':
            MT_GF_FILE = '/Users/kcoppess/muspelheim/synthetic-seismograms/synthetic-seismograms/greens_functions/halfspace/halfA_conduit/extended_1km_mt/'
else:
    MT_GF_FILE = args.MTGF

if args.SFGF == 'see main.py':
    if SOURCE_TYPE == 'CHAMBER':
        SF_GF_FILE = '/Users/kcoppess/muspelheim/synthetic-seismograms/synthetic-seismograms/greens_functions/halfspace/halfA_chamber/halfA_1.028794_sf/'
    elif SOURCE_TYPE == 'CONDUIT':
        if REPRESENTATION == 'PS':
            SF_GF_FILE = '/Users/kcoppess/muspelheim/synthetic-seismograms/synthetic-seismograms/greens_functions/halfspace/halfA_conduit/halfA_0.50195_sf/'
        elif REPRESENTATION == 'ES':
            SF_GF_FILE = '/Users/kcoppess/muspelheim/synthetic-seismograms/synthetic-seismograms/greens_functions/halfspace/halfA_conduit/extended_1km_sf/'
else:
    SF_GF_FILE = args.SFGF

if args.sourcedepth < 0:
    if SOURCE_TYPE == 'CONDUIT':
        point = 501.95  # m
    elif SOURCE_TYPE == 'CHAMBER':
        point = 1028.794  # m
else:
    point = args.sourcedepth

chamber_vol = args.chamvol
conduit_radius = args.condrad

gc.collect()


TIME_INPUT = ''  # MANUAL or anything else (anything other than MANUAL draws data from the file directory)
MOMENT_PRESSURE = [0, 1, 2]  # moment pressure time series for manual entry
FORCE_PRESSURE = [0, 1, 2]  # force pressure time series for manual entry
MANUAL_TIME = [0, 1, 2]  # time for time series for manual entry


if REPRESENTATION == 'PS':
    DEPTH = str(point * 1e-3)+' km'
else:
    DEPTH = ''

'''------------------------------------------------------------------------------------------'''
'''
receiver/seismometer specs:
spatial coordinate labels
nn seismometers with (x,y,z) position
'''
labels = np.loadtxt(args.stations, dtype=str, comments='#', delimiter=' ', usecols=0)
pos = np.loadtxt(args.stations, dtype=float, comments='#', delimiter=' ', usecols=(1,2,3))
nn = len(labels)

'''------------------------------------------------------------------------------------------'''

'''medium parameters set'''
v_s = 2000 # m/s
v_p = 3464.1016 # m/s
# density
rho_rock = 2700  # kg/m^3
# shear modulus (when mu = 0, just have p-waves and matches acoustic)
mu = rho_rock * v_s**2  # Pa
# p-wave modulus
Kp = rho_rock * v_p**2  # Pa
# Lame parameter
lame = Kp - 2 * mu

## bulk modulus
#K = 5 * mu / 3  # for a poisson ratio = 1/4 
## Lame parameter
#lame = K - (2/3)*mu

'''source description'''
'''conduit'''
# cross-sectional area of the tube (assuming constant cross-section throughout) (m^2)
A = np.pi * conduit_radius**2

'''for chamber'''
# scaled volume of magma chamber (assuming spherical)
vol = (3/4) * chamber_vol  # =np.pi * chamber_radius**3

if SOURCE_TYPE == 'CONDUIT':
    sourceDim = A
elif SOURCE_TYPE == 'CHAMBER':
    if CONTRIBUTION == 'MOMENT':
        sourceDim = vol
    if CONTRIBUTION == 'FORCE':
        sourceDim = A
sourcePos = np.array([0,0,-point])


'''------------------------------------------------------------------------------------------'''
if CONTRIBUTION == 'MOMENT' or CONTRIBUTION == 'BOTH':
    p, time, height = ld.moment_ZIP_load(zip_filename, SOURCE_TYPE, TOTAL_TIME, DT)
    print('loaded data...')
    if TIME_INPUT == 'MANUAL':  # option to manually set moment time series
        p = MOMENT_PRESSURE
        time = MANUAL_TIME
    if REPRESENTATION == 'PS':
        r_mom, z_mom, tr_mom, moment = PS.moment_general(SOURCE_TYPE, p, height, time, pos, labels, 
                                                [sourceDim, sourcePos], [mu, lame, rho_rock], MT_GF_FILE, deriv=DERIV,
                                                INTERPOLATE=True, SOURCE_FILTER=True)
    elif REPRESENTATION == 'ES':
        r_mom, z_mom, tr_mom, moment = ES.moment_general(p, np.flip(height), time, pos, labels, 
                                                [sourceDim, sourcePos], [mu, lame, rho_rock], MT_GF_FILE, deriv=DERIV,
                                                INTERPOLATE=True, SOURCE_FILTER=True, SAVES=False, mt_savefile=MT_GF_FILE)

    gc.collect()
    if SAVE:
        np.savetxt(save_file+'MOMENT.gz', moment, delimiter=',')
        for ii, LAB in zip(range(nn), labels):
            np.savetxt(save_file+'MOMENT__r'+LAB+'_radial.gz', r_mom[ii], delimiter=',')
            np.savetxt(save_file+'MOMENT__r'+LAB+'_vertical.gz', z_mom[ii], delimiter=',')
            np.savetxt(save_file+'MOMENT__r'+LAB+'_transverse.gz', tr_mom[ii], delimiter=',')
if CONTRIBUTION == 'FORCE' or CONTRIBUTION == 'BOTH':
    f, time, height = ld.force_ZIP_load(zip_filename, SOURCE_TYPE, TOTAL_TIME, DT)
    if TIME_INPUT == 'MANUAL':  # option to manually set force time series
        f = FORCE_PRESSURE
        time = MANUAL_TIME
    if REPRESENTATION == 'PS':
        r_for, z_for, tr_for, force = PS.force_general(SOURCE_TYPE, f, height, time, pos, labels, 
                                                [sourceDim, sourcePos], [mu, lame, rho_rock], SF_GF_FILE, deriv=DERIV,
                                                INTERPOLATE=True, SOURCE_FILTER=True)
    elif REPRESENTATION == 'ES':
        r_for, z_for, tr_for, force = ES.force_general(f, np.flip(height), time, pos, labels, 
                                                [sourceDim, sourcePos], [mu, lame, rho_rock], SF_GF_FILE, deriv=DERIV,
                                                INTERPOLATE=True, SOURCE_FILTER=True, SAVES=False, sf_savefile=SF_GF_FILE)
    gc.collect()
    if SAVE:
        np.savetxt(save_file+'FORCE.gz', force, delimiter=',')
        for ii, LAB in zip(range(nn), labels):
            np.savetxt(save_file+'FORCE__r'+LAB+'_radial.gz', r_for[ii], delimiter=',')
            np.savetxt(save_file+'FORCE__r'+LAB+'_vertical.gz', z_for[ii], delimiter=',')
            np.savetxt(save_file+'FORCE__r'+LAB+'_transverse.gz', tr_for[ii], delimiter=',')
if SAVE:
    np.savetxt(direc+'TIME.gz', time, delimiter=',')
gc.collect()

r = np.zeros((nn, len(time)), dtype='complex')
z = np.zeros((nn, len(time)), dtype='complex')
tr = np.zeros((nn, len(time)), dtype='complex')

if CONTRIBUTION == 'MOMENT' or CONTRIBUTION == 'BOTH':
    r += r_mom
    gc.collect()
    z += z_mom
    gc.collect()
    tr += tr_mom
    gc.collect()
if CONTRIBUTION == 'FORCE' or CONTRIBUTION == 'BOTH':
    r += r_for
    gc.collect()
    z += z_for
    gc.collect()
    tr += tr_for
    gc.collect()


print(SOURCE_TYPE+' '+CONTRIBUTION+' '+REPRESENTATION+' '+DERIV)

'''plotting combined waveform'''
if PLOT:
    colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000']
    line_styles = ['-', '--', ':', '-.', '.']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False, figsize=(10,10))
    ax1, ax2, ax3 = hp.seismogram_plot_setup([ax1, ax2, ax3], 'single-source: '+SOURCE_TYPE+' '+CONTRIBUTION+' '+DEPTH)

    for ii in range(nn):
        ax1.plot(time, r[ii], color=colors[ii], linestyle=line_styles[0], label=labels[ii], linewidth=1.5)
    ax1.set_ylabel('radial ($r$)')
    ax1.set_xlim(0, np.max(time))
    ax1.legend()

    for ii in range(nn):
        ax2.plot(time, tr[ii], color=colors[ii], linestyle=line_styles[0], linewidth=1.5)
    ax2.set_ylabel('transverse ($\phi$)')

    for ii in range(nn):
        ax3.plot(time, z[ii], color=colors[ii], linestyle=line_styles[0], linewidth=1.5)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('vertical ($\\theta$)')
    plt.show()
