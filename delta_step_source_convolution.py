import numpy as np
import scipy.integrate as si
import load_gfs as gf
import helpers as hp
import matplotlib.pyplot as plt
import gc
import source_setup as ss

#SOURCE = 'DELTA'
SOURCE = 'STEP'

gf_file = 'greens_functions/halfspace/halfA_conduit/halfA_0.50195_'
#gf_file = 'greens_functions/halfspace/halfA_chamber/halfA_1.133650_'
source_depth = 501.95 #1133.650 #501.95
conduit_rad = 20 #m
conduit_len = 1000 #m
stations = ['10km']
stat_dist = [10000]
colors = ['#56B4E9']
#stations = ['1km', '3km', '10km', '30km']
#stat_dist = [1000, 3000, 10000, 30000]
#colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000']

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

pArrival = np.sqrt(source_depth**2 + stat_dist[0]**2) / v_p
sArrival = np.sqrt(source_depth**2 + stat_dist[0]**2) / v_s
RArrival = np.sqrt(source_depth**2 + stat_dist[0]**2) / (0.92 * v_s)

dt = 0.04
source_time = np.arange(500) * dt

nn = len(stations)
tt = len(source_time)

if SOURCE == 'STEP':
    sig = 0.1
    time_shift = 6 * sig
    force_rate = np.exp(-((source_time - time_shift)/ sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)
    
    omega = np.fft.fftfreq(tt, dt) * (2 * np.pi)
    
    force_rate_hat = np.fft.fft(force_rate) * dt
    force_rate_hat *= np.exp(1j * omega * time_shift)
    force = si.cumtrapz(force_rate, x=source_time, initial=0)
elif SOURCE == 'DELTA':
    sig = 0.1
    time_shift = 6 * sig
    force = np.exp(-((source_time - time_shift)/ sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)
    
    omega = np.fft.fftfreq(tt, dt) * (2 * np.pi)
    force_rate = np.gradient(force, source_time)
    
    force_rate_hat = np.fft.fft(force_rate) * dt
    force_rate_hat *= np.exp(1j * omega * time_shift)


z_for = np.zeros((nn,tt), dtype='complex')
r_for = np.zeros((nn,tt), dtype='complex')
tr_for = np.zeros((nn,tt), dtype='complex')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False)
ax1.set_title('single force')

ax1.axvline(pArrival, alpha=0.2)
ax2.axvline(pArrival, alpha=0.2)
ax3.axvline(pArrival, alpha=0.2)
ax1.axvline(sArrival, alpha=0.2)
ax2.axvline(sArrival, alpha=0.2)
ax3.axvline(sArrival, alpha=0.2)
ax1.axvline(RArrival, alpha=0.2)
ax2.axvline(RArrival, alpha=0.2)
ax3.axvline(RArrival, alpha=0.2)

for stat, ii in zip(stations, np.arange(nn)):
    time, gfs = gf.load_gfs_PS(gf_file+'sf/'+stat+'/', 1, source_time, INTERPOLATE_TIME=True, SAVE=False, PLOT=False)
    
    gfs_hat = []
    for gg in gfs:
        gf_hat = np.fft.fft(gg, axis=0) * dt
        gfs_hat.append(gf_hat)
    z_for[ii] += si.cumtrapz(np.fft.ifft(force_rate_hat * gfs_hat[1][:,0], axis=-1) / dt, x=source_time, initial=0)
    r_for[ii] += si.cumtrapz(np.fft.ifft(force_rate_hat * gfs_hat[1][:,1], axis=-1) / dt, x=source_time, initial=0)
    tr_for[ii] += si.cumtrapz(np.fft.ifft(force_rate_hat * gfs_hat[1][:,2], axis=-1) / dt, x=source_time, initial=0)
    
    ax1.plot(source_time, np.real(z_for[ii]), alpha=0.7, color=colors[ii], label=stat)
    ax2.plot(source_time, np.real(r_for[ii]), alpha=0.7, color=colors[ii])

ax1.set_ylabel('vertical (m)')
ax2.set_ylabel('radial (m)')
ax3.set_xlabel('time (s)')
ax1.legend()

ax3.plot(source_time, force)
ax3.set_ylabel('force history (N)')
plt.tight_layout()
plt.show()


# for cylindrical source

if SOURCE == 'STEP':
    pressure_rate = 1.0 * np.exp(-((source_time - time_shift) / sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)
    
    pressure = si.cumtrapz(pressure_rate, x=source_time, initial=0)
    moment_rate = ss.moment_density(np.array([pressure_rate]), np.pi * conduit_rad**2 * conduit_len, cushion=0)[0]
    moment_tensor = ss.moment_tensor_cylindricalSource([lame, mu])
    moment = si.cumtrapz(moment_rate, x=source_time, initial=0)
    
    general_MT_hat = np.fft.fft(moment_rate, axis=-1) * dt
    general_MT_hat *= np.exp(1j * omega * time_shift)
elif SOURCE == 'DELTA':
    pressure = 1.0 * np.exp(-((source_time - time_shift) / sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)
    pressure_rate = np.gradient(pressure, source_time)
    
    moment_rate = ss.moment_density(np.array([pressure_rate]), np.pi * conduit_rad**2 * conduit_len, cushion=0)[0]
    moment_tensor = ss.moment_tensor_cylindricalSource([lame, mu])
    moment = si.cumtrapz(moment_rate, x=source_time, initial=0)
    
    general_MT_hat = np.fft.fft(moment_rate, axis=-1) * dt
    general_MT_hat *= np.exp(1j * omega * time_shift)

z_mom = np.zeros((nn,tt), dtype='complex')
r_mom = np.zeros((nn,tt), dtype='complex')
tr_mom = np.zeros((nn,tt), dtype='complex')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = False)
ax1.set_title('moment tensor')

ax1.axvline(pArrival, alpha=0.2)
ax2.axvline(pArrival, alpha=0.2)
ax3.axvline(pArrival, alpha=0.2)
ax1.axvline(sArrival, alpha=0.2)
ax2.axvline(sArrival, alpha=0.2)
ax3.axvline(sArrival, alpha=0.2)
ax1.axvline(RArrival, alpha=0.2)
ax2.axvline(RArrival, alpha=0.2)
ax3.axvline(RArrival, alpha=0.2)

for stat, ii in zip(stations, np.arange(nn)):
    time, gfs = gf.load_gfs_PS(gf_file+'mt/'+stat+'/', 0, source_time, INTERPOLATE_TIME=True, SAVE=False, PLOT=False)
    gfs_hat = []
    for gg in gfs:
        gf_hat = np.fft.fft(gg, axis=0) * dt
        gfs_hat.append(gf_hat)
    # ['Mxx.txt', '2Mxy.txt', '2Mxz.txt', 'Myy.txt', '2Myz.txt', 'Mzz.txt']

    #plt.plot(source_time, moment_tensor_rate[0,0])
    #plt.plot(source_time, gfs[0][:,0], label='vertical')
    #plt.show()

    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,0] * gfs_hat[0][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,1] * gfs_hat[1][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,2] * gfs_hat[2][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[1,1] * gfs_hat[3][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[1,2] * gfs_hat[4][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[2,2] * gfs_hat[5][:,0], axis=-1) / dt, x=source_time, initial=0)
    gc.collect()

    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,0] * gfs_hat[0][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,1] * gfs_hat[1][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,2] * gfs_hat[2][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[1,1] * gfs_hat[3][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[1,2] * gfs_hat[4][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[2,2] * gfs_hat[5][:,1], axis=-1) / dt, x=source_time, initial=0)
    gc.collect()

    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,0] * gfs_hat[0][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,1] * gfs_hat[1][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[0,2] * gfs_hat[2][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[1,1] * gfs_hat[3][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[1,2] * gfs_hat[4][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat * moment_tensor[2,2] * gfs_hat[5][:,2], axis=-1) / dt, x=source_time, initial=0)
    gc.collect()

    ax1.plot(source_time, np.real(z_mom[ii]), alpha=0.7, color=colors[ii], label=stat)
    ax2.plot(source_time, np.real(r_mom[ii]), alpha=0.7, color=colors[ii])
ax1.set_ylabel('vertical (m)')
ax2.set_ylabel('radial (m)')
ax3.set_xlabel('time (s)')
ax1.legend()

ax3.plot(source_time, moment * moment_tensor[2, 2])
ax3.set_ylabel('moment $M_{zz}$ (Nm)')
plt.tight_layout()
plt.show()
