#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import gc

import synthetic as syn
import helpers as hp


# In[ ]:


'''loading data and parameters'''
directory = '/Users/kcoppess/muspelheim/kilauea_collapse/'
data = np.loadtxt(directory+'pressure_stress_history.txt', delimiter=',', skiprows=1).transpose()
data1 = np.loadtxt(directory+'resonance_history.txt', delimiter=',', skiprows=1).transpose()

time = np.array(data[0])
pressure = np.array(data[1]) + np.array(data1[1])
shear_force = -np.array(data[2]) # positive upwards

# origin is located at lost accelerometer location
accel_pos_xy = np.loadtxt(directory+'acce_local_xy.txt', delimiter=',', skiprows=1)

# accelerometer/receiver location w/ origin NPIT
accel_pos = np.zeros((5, 3))
accel_pos[:,:2] = accel_pos_xy

accel_labels = ['HMLE', 'NPT', 'PAUD', 'RSDD', 'UWE']

parameters = np.loadtxt(directory+'parameters.txt', delimiter=',', skiprows=1)

chamber_cent = np.array(parameters[:3])
chamber_cent[2] = -chamber_cent[2] # z positive upwards

piston_cent = np.array(parameters[:3])
chamber_top_depth = abs(parameters[3])
piston_cent[2] = -parameters[3] * 0.75 # z positive upwards (placing source about 0.75 chamber top depth)
piston_radius = parameters[5]


# In[ ]:


'''medium parameters set'''
# density
rho_rock = 2700 # kg/m^3
# shear modulus (when mu = 0, just have p-waves and matches acoustic)
mu = 3e9 # Pa

dt = time[2] - time[1]

chamber_vol = 3e9 # m^3
chamberParams = [chamber_vol, chamber_cent]

pistonParams = [chamber_top_depth, piston_radius, piston_cent]


x, y, z = syn.synthetic(pressure, shear_force, dt, accel_pos, chamberParams, pistonParams, [mu, rho_rock], deriv='DIS')


# In[ ]:


'''plotting combined waveform'''

nn = 5

colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000']
line_styles = ['-', '--', ':', '-.', '.']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False, figsize=(10,10))
ax1, ax2, ax3 = hp.seismogram_plot_setup([ax1, ax2, ax3], 'single-source:')

for ii in range(nn):
    ax1.plot(time, x[ii], color=colors[ii], linestyle=line_styles[0], label=accel_labels[ii], linewidth=1.5)
ax1.set_ylabel('radial ($r$)')
ax1.set_xlim(-18, np.max(time))
ax1.legend()
# ax1.set_ylim(-1.5e-10, 1e-10)
#ax1.set_ylim(-0.00003, 0.00003)

for ii in range(nn):
    ax2.plot(time, y[ii], color=colors[ii], linestyle=line_styles[0], linewidth=1.5)
ax2.legend(loc=1, fontsize='small')
ax2.set_ylabel('transverse ($\phi$)')

for ii in range(nn):
    ax3.plot(time, z[ii], color=colors[ii], linestyle=line_styles[0], linewidth=1.5)
ax3.set_xlabel('time (s)')
ax3.set_ylabel('vertical ($\\theta$)')

# plt.savefig(plot_file+'PS_far_'+str(int(rr / 1000))+'km.png', dpi = 300)

plt.show()


# In[ ]:




