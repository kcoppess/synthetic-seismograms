import numpy as np
import scipy.integrate as si
import load_gfs as gf
import helpers as hp
import matplotlib.pyplot as plt
import gc

def sf_static_displacement(P, r, x):
    '''
    static displacement from single force P (+ upward) (Mindlin, 1936)

    returns radial U (+ out) and vertical w (+ up)
    '''
    G = 3e9 #2700 * 2000**2 #Pa
    mu = 0.25 #0.33333333333333 #0.25 #poisson ratio
    z = 0 # receiver depth
    
    R1 = (r**2 + (z - x)**2)**0.5
    R2 = (r**2 + (z + x)**2)**0.5
    U = (-P*r/(16*np.pi*G*(1-mu)))*(((z-x)/R1**3)+((3-4*mu)*(z-x)/R2**3)-(4*(1-mu)*(1-2*mu)/(R2*(R2+z+x)))+(6*x*z*(z+x)/R2**5))
    w = (P/(16*np.pi*G*(1-mu)))*(((3-4*mu)/R1)+((8*(1-mu)**2 - (3-4*mu))/R2) + ((z-x)**2/R1**3) + (((3-4*mu)*(z+x)**2 -2*x*z)/R2**3) + ((6*x*z*(z+x)**2)/R2**5))

    return U, w

def mt_static_displacement(P, a, f, R):
    '''
    static surface displacement from spherical pressure point source (Mogi, 1958)

    returns radial U (+ out) and vertical w (+ up)
    '''
    mu = 3e9 #Pa
    
    A = 3 * a**3 * P / (4 * mu)
    R1 = (f**2 + R**2)**(3/2)
    U = A * R / R1
    w = A * f / R1

    return U, w

depths = np.linspace(0, 20000, 10000)
U, w = sf_static_displacement(1, 1000, depths)

UU = []

for dd in [1000, 3000, 10000, 30000]:
    U1, w1 = sf_static_displacement(1, dd, 501.95)
    UU.append(w1)
print(UU)

plt.axhline(0, alpha=0.3)
plt.plot(depths, U, label='radial')
plt.plot(depths, w, label='vertical')
plt.show()

'''
gf_file = 'greens_functions/halfspace_Kilauea/'
stations = ['HMLE', 'NPT', 'PAUD', 'RSDD', 'UWE']
stat_dist = [6328.43188889,  478.10014591, 7608.27249803, 5674.95589884, 1939.06594614]
colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000']

dt = 0.1
source_time = np.arange(300) * dt
print(len(source_time))

nn = len(stations)
tt = len(source_time)

sig = 0.1
time_shift = 6 * sig
force_rate = np.exp(-((source_time - time_shift)/ sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)

omega = np.fft.fftfreq(tt, dt) * (2 * np.pi)

force_rate_hat = np.fft.fft(force_rate) * dt
force_rate_hat *= np.exp(1j * omega * time_shift)
#force = si.cumtrapz(force_rate, x=source_time, initial=0)

#plt.plot(source_time, force)
#plt.show()


z_for = np.zeros((nn,tt), dtype='complex')
r_for = np.zeros((nn,tt), dtype='complex')
tr_for = np.zeros((nn,tt), dtype='complex')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey = False)
ax1.set_title('single force')

for stat, ii in zip(stations, np.arange(nn)):
    time, gfs = gf.load_gfs(gf_file+stat+'/', 1, source_time, INTERPOLATE=True, SAVE=False, PLOT=False)
    
    #plt.plot(source_time, force_rate)
    #plt.plot(source_time, gfs[0][:, 0], label='vertical')
    #plt.plot(source_time, gfs[1][:, 1], label='radial')
    #plt.legend()
    #plt.show()
    
    gfs_hat = []
    for gg in gfs:
        gf_hat = np.fft.fft(gg, axis=0) * dt
        gfs_hat.append(gf_hat)
    z_for[ii] += si.cumtrapz(np.fft.ifft(force_rate_hat * gfs_hat[1][:,0], axis=-1) / dt, x=source_time, initial=0)
    r_for[ii] += si.cumtrapz(np.fft.ifft(force_rate_hat * gfs_hat[1][:,1], axis=-1) / dt, x=source_time, initial=0)
    tr_for[ii] += si.cumtrapz(np.fft.ifft(force_rate_hat * gfs_hat[1][:,2], axis=-1) / dt, x=source_time, initial=0)
    
    U, w = sf_static_displacement(1, stat_dist[ii], 556.54187)
    print(stat)
    print(r_for[ii, -1], z_for[ii, -1])
    print(U, w)
    print('-----')

    ax1.axhline(w, color=colors[ii], alpha=0.3)
    ax1.plot(source_time, z_for[ii], alpha=0.7, color=colors[ii], label=stat)
    ax2.axhline(U, color=colors[ii], alpha=0.3)
    ax2.plot(source_time, r_for[ii], alpha=0.7, color=colors[ii])
ax1.set_ylabel('vertical (m)')
ax2.set_ylabel('radial (m)')
ax2.set_xlabel('time (s)')
ax1.legend()
plt.show()

chamber_vol = 4e9 #m^3
chamber_aspc = 1

pressure_rate = 6.923e6 * np.exp(-((source_time - time_shift) / sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)
ra = int((chamber_aspc**2 * chamber_vol * 3 / (4*np.pi))**(1/3)) # semi-major axis length
rb = int(ra/chamber_aspc)                                        # semi-minor axis length

pressure = si.cumtrapz(pressure_rate, x=source_time, initial=0)
#plt.plot(source_time, pressure/6.923e6)
#plt.show()

moment_tensor_rate = spmt.sphr_MT(rb, rb, ra, pressure_rate, [0.25, 3e9])
moment_tensor_rate = hp.diag(moment_tensor_rate)

moment_tensor = si.cumtrapz(moment_tensor_rate, x=source_time, initial=0)
plt.plot(source_time, moment_tensor[0,0])
plt.plot(source_time, moment_tensor[1,1])
plt.plot(source_time, moment_tensor[2,2])
plt.show()

general_MT_hat = np.fft.fft(moment_tensor_rate, axis=2) * dt
general_MT_hat *= np.exp(1j * omega * time_shift)

z_mom = np.zeros((nn,tt), dtype='complex')
r_mom = np.zeros((nn,tt), dtype='complex')
tr_mom = np.zeros((nn,tt), dtype='complex')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey = False)
ax1.set_title('moment tensor')

for stat, ii in zip(stations, np.arange(nn)):
    time, gfs = gf.load_gfs(gf_file+stat+'/', 0, source_time, INTERPOLATE=True, SAVE=False, PLOT=False)
    gfs_hat = []
    for gg in gfs:
        gf_hat = np.fft.fft(gg, axis=0) * dt
        gfs_hat.append(gf_hat)
    # ['Mxx.txt', '2Mxy.txt', '2Mxz.txt', 'Myy.txt', '2Myz.txt', 'Mzz.txt']

    #plt.plot(source_time, moment_tensor_rate[0,0])
    #plt.plot(source_time, gfs[0][:,0], label='vertical')
    #plt.show()

    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,1] * gfs_hat[1][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,2] * gfs_hat[2][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,2] * gfs_hat[4][:,0], axis=-1) / dt, x=source_time, initial=0)
    z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,0], axis=-1) / dt, x=source_time, initial=0)
    gc.collect()

    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,1] * gfs_hat[1][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,2] * gfs_hat[2][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,2] * gfs_hat[4][:,1], axis=-1) / dt, x=source_time, initial=0)
    r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,1], axis=-1) / dt, x=source_time, initial=0)
    gc.collect()

    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,1] * gfs_hat[1][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,2] * gfs_hat[2][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,2] * gfs_hat[4][:,2], axis=-1) / dt, x=source_time, initial=0)
    tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,2], axis=-1) / dt, x=source_time, initial=0)
    gc.collect()

    U, w = mt_static_displacement(6.923e6, ra, 2177.29975, stat_dist[ii])
    print(stat)
    print(r_mom[ii, -1], z_mom[ii, -1])
    print(U, w)
    print('-----')
    
    ax1.axhline(w, color=colors[ii], alpha=0.3)
    ax1.plot(source_time, z_mom[ii], alpha=0.7, color=colors[ii], label=stat)
    ax2.axhline(U, color=colors[ii], alpha=0.3)
    ax2.plot(source_time, r_mom[ii], alpha=0.7, color=colors[ii])
ax1.set_ylabel('vertical (m)')
ax2.set_ylabel('radial (m)')
ax2.set_xlabel('time (s)')
ax1.legend()
plt.show()
'''
