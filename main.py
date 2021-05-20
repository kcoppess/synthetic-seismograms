#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import gc
import os
from zipfile import ZipFile
import scipy.signal as sg
import scipy.interpolate as si

import point_source as PS
import synthetic as syn
import helpers as hp
import source_setup as ss


SIMULATION = 'sharper__RUPTURE__100s_128pts__CHAM_10e6m3__PLUG_02e6Pa_1e-03_pos0750m__MAGMA_cVF80_n001'
SOURCE_TYPE = 'CONDUIT'
CONTRIBUTION = 'MOMENT'
TOTAL_TIME = 100 #in seconds
SAVE = True
PLOT = False

conduit_radius = 30 #m
chamber_vol = 1e5 #m^3

# source depth
if SOURCE_TYPE == 'CONDUIT':
    point = 500 #m
elif SOURCE_TYPE == 'CHAMBER':
    point = 1030 #m

'''------------------------------------------------------------------------------------------'''
'''receiver/seismometer specs'''
# number of seismometers
nn = 1
# all seismometers equal distant from origin (m)
rr = 1e3

'''
spatial coordinate labels
nn seismometers with (x,y,z) position
fixing seismometers to y-z plane cutting through the source at origin (x = 0)
'''
labels = ['pi/2', 'pi/3', 'pi/4', 'pi/6', '0']
# unit vector positions for each seismometer
pos_unit = np.array([[0, 1, 0]])#,
#                      [0, np.sqrt(3)/2, -0.5],
#                      [0, np.sqrt(0.5), -np.sqrt(0.5)],
#                      [0, 0.5, -np.sqrt(3)/2],
#                      [0, np.sqrt(1-0.9999999999999**2), -0.9999999999999]])
pos = pos_unit * rr

'''------------------------------------------------------------------------------------------'''

'''medium parameters set'''
# density
rho_rock = 2700 # kg/m^3
# shear modulus (when mu = 0, just have p-waves and matches acoustic)
mu = 5e9 # Pa
# bulk modulus
K = 5 * mu / 3 # for a poisson ratio = 1/4
# Lame parameter
lame = K - (2/3)*mu

'''source description'''
'''conduit'''
# cross-sectional area of the tube (assuming constant cross-section throughout) (m^2)
A = np.pi * conduit_radius**2

'''for chamber'''
# scaled volume of magma chamber (assuming spherical)
vol = (3/4) * chamber_vol #np.pi * chamber_radius**3

if SOURCE_TYPE == 'CONDUIT':
    sourceDim = A
elif SOURCE_TYPE == 'CHAMBER':
    sourceDim = vol
sourcePos = np.array([0,0,-point])


'''------------------------------------------------------------------------------------------'''
'''loading data and setting up directories'''

direc = '/Users/kcoppess/muspelheim/synthetic-seismograms/seismos/'+SIMULATION+'/'
save_file = direc+SOURCE_TYPE+'__'
if not os.path.exists(direc):
    os.makedirs(direc)

directory = '/Users/kcoppess/muspelheim/simulation-results/plug_rupture/'+SIMULATION+'__'
zip_filename = directory+'.zip'

if CONTRIBUTION == 'MOMENT' or CONTRIBUTION == 'BOTH':
    p, time, height = PS.moment_ZIP_load(zip_filename, SOURCE_TYPE, TOTAL_TIME)
    dt = time[2] - time[1]
    x_mom, y_mom, z_mom, moment = PS.moment_synthetic(SOURCE_TYPE, p, height, dt, pos, [sourceDim, sourcePos], 
                                            [mu, rho_rock], WAVE='BOTH')
    if SAVE:
        np.savetxt(save_file+'MOMENT__r'+str(int(rr))+'_x.gz', x_mom, delimiter=',')
        np.savetxt(save_file+'MOMENT__r'+str(int(rr))+'_y.gz', y_mom, delimiter=',')
        np.savetxt(save_file+'MOMENT__r'+str(int(rr))+'_z.gz', z_mom, delimiter=',')
        np.savetxt(save_file+'MOMENT.gz', moment, delimiter=',')
elif CONTRIBUTION == 'FORCE' or CONTRIBUTION == 'BOTH':
    f, time, height = PS.force_ZIP_load(zip_filename, SOURCE_TYPE, TOTAL_TIME)
    dt = time[2] - time[1]
    x_for, y_for, z_for, force = PS.force_synthetic(SOURCE_TYPE, f, height, dt, pos, [A, sourcePos], 
                                            [mu, rho_rock], WAVE='BOTH')
    if SAVE:
        np.savetxt(save_file+'FORCE__r'+str(int(rr))+'_x.gz', x_for, delimiter=',')
        np.savetxt(save_file+'FORCE__r'+str(int(rr))+'_y.gz', y_for, delimiter=',')
        np.savetxt(save_file+'FORCE__r'+str(int(rr))+'_z.gz', z_for, delimiter=',')
        np.savetxt(save_file+'FORCE.gz', force, delimiter=',')

x = np.zeros((nn, len(time)), dtype='complex')
y = np.zeros((nn, len(time)), dtype='complex')
z = np.zeros((nn, len(time)), dtype='complex')

if CONTRIBUTION == 'MOMENT' or CONTRIBUTION == 'BOTH':
    x += x_mom
    gc.collect()
    y += y_mom
    gc.collect()
    z += z_mom
    gc.collect()
elif CONTRIBUTION == 'FORCE' or CONTRIBUTION == 'BOTH':
    x += x_for
    gc.collect()
    y += y_for
    gc.collect()
    z += z_for
    gc.collect()

'''
transforming x, y, z waveforms into the radial, vertical, and transverse waveforms
NOTE: vertical from spherical is positive "downwards"
                  cylindrical is positive "upwards"
'''
NN = len(time)
rad = np.zeros((nn, NN), dtype = 'complex')
tra = np.zeros((nn, NN), dtype = 'complex')
ver = np.zeros((nn, NN), dtype = 'complex')
for ii in range(nn):
    rad[ii,:], tra[ii,:], ver[ii,:] = hp.cartesian_to_cylindrical(x[ii], y[ii], z[ii], pos[ii])

gc.collect()
# In[ ]:


'''plotting combined waveform'''
if PLOT:
    colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000']
    line_styles = ['-', '--', ':', '-.', '.']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False, figsize=(10,10))
    ax1, ax2, ax3 = hp.seismogram_plot_setup([ax1, ax2, ax3], 'single-source:')
    
    for ii in range(nn):
        ax1.plot(time, rad[ii], color=colors[ii], linestyle=line_styles[0], label=labels[ii], linewidth=1.5)
    ax1.set_ylabel('radial ($r$)')
    ax1.set_xlim(0, np.max(time))
    ax1.legend()
    # ax1.set_ylim(-1.5e-10, 1e-10)
    #ax1.set_ylim(-0.00003, 0.00003)
    
    for ii in range(nn):
        ax2.plot(time, tra[ii], color=colors[ii], linestyle=line_styles[0], linewidth=1.5)
    ax2.legend(loc=1, fontsize='small')
    ax2.set_ylabel('transverse ($\phi$)')
    
    for ii in range(nn):
        ax3.plot(time, ver[ii], color=colors[ii], linestyle=line_styles[0], linewidth=1.5)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('vertical ($\\theta$)')
    
    # plt.savefig(plot_file+'PS_far_'+str(int(rr / 1000))+'km.png', dpi = 300)
    
    plt.show()


