import numpy as np
import gc
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.io as sio
import scipy.integrate as sint

def load_gfs_PS(directory, srctype, time, INTERPOLATE_TIME=False, SAVE=False, save_file='gf', PLOT=False, UU=0):
    '''
    loads in point source Green's functions and can interpolate in time to get compatible
    array dimensions with desired time array
    can also save the Green's functions to a new file 
    
    NB: if don't want to interpolate each time, can save the resulting interpolated 
        Green's functions and just load them in next time without alteration
    
    original Green's functions must be stored such that the columns correspond below:
            time    vertical    radial   transverse

    --INPUTS--
    directory        : string            : path to folder holding Green's function files
    srctype          : (1)               : source type (0: moment tensor, 1: single force)
    time             : (# time points)   : desired time array
    INTERPOLATE_TIME : bool              : if True, interpolate to get values at desired time
    SAVE             : bool              : if True, saves Green's functions to save_file
    save_file        : string            : path to where Green's functions will be saved
    --RETURNS--
    gf_time     : (# time points)        : desired time array
    gfs         : [ (# time points, 3) ] : list of final Green's functions (ver, rad, tra)
                                            if single force, 2 arrays
                                            if moment tensor, 6 arrays
    '''
    if srctype == 0:
        components = ['Mxx.txt', '2Mxy.txt', '2Mxz.txt', 'Myy.txt', '2Myz.txt', 'Mzz.txt']
    elif srctype == 1:
        components = ['horizontal_force.txt', 'vertical_force.txt']
    colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000', '#E50000']
    
    gfs = []
    gfs_hat = []
    gf_time = np.loadtxt(directory+components[0], usecols = 0)
    gf_dt = gf_time[1] - gf_time[0]
    for com in components:
        gf = np.loadtxt(directory+com, usecols = (1,2,3))
        gfs.append(gf)
        gfs_hat.append(np.fft.fft(gf, axis=0) * gf_dt)
    gc.collect()

    gf_omega = np.fft.fftfreq(len(gf_time), gf_dt) * (2 * np.pi)
    gf_ind = np.argwhere(gf_omega < 0)[0,0]
    sorted_gf_omega = np.concatenate((gf_omega[gf_ind:], gf_omega[:gf_ind]))
    gc.collect()

    '''-------------------------------------------------------------------------------------'''
    sig = 0.1
    time_shift = 6 * sig
    force_rate = np.exp(-((gf_time - time_shift)/ sig) **2 / 2) / (np.sqrt(2 * np.pi) * sig)
    
    force_rate_hat = np.fft.fft(force_rate) * gf_dt
    force_rate_hat *= np.exp(1j * gf_omega * time_shift)
    '''-------------------------------------------------------------------------------------'''
    
    if PLOT:
        plt.figure()
        plt.title(directory[-5:])
        plt.axhline(UU, alpha=0.5)
        for func, lab, col in zip(gfs_hat, components, colors):
            #plt.plot(gf_omega, np.abs(func[:,1]), color=col, label=lab)
            #plt.plot(gf_time, func[:,1], color=col, label=lab)
            #plt.plot(gf_time, sint.cumtrapz(np.fft.ifft(func[:,1], axis=0) / gf_dt, x=gf_time, initial=0), color=col, label=lab) 
            plt.plot(gf_time, np.fft.ifft(func[:,1], axis=0) / gf_dt, color=col, label=lab) 
        #plt.legend()
        #plt.show()
    new_gfs = []
    if INTERPOLATE_TIME:
        tt = len(time)
        dt = time[2] - time[1]
        desired_omega = np.fft.fftfreq(tt, dt) * (2 * np.pi)
        ind = np.argwhere(desired_omega < 0)[0,0]
        sorted_desired_omega = np.concatenate((desired_omega[ind:], desired_omega[:ind]))
        for func, lab, col in zip(gfs_hat, components, colors):
            sorted_func = np.concatenate((func[gf_ind:], func[:gf_ind]))
            #smoothreal = si.Akima1DInterpolator(sorted_gf_omega, np.real(sorted_func), axis=0)
            #smoothimag = si.Akima1DInterpolator(sorted_gf_omega, np.imag(sorted_func), axis=0)
            #sorted_gf_hat_sm = smoothreal(sorted_desired_omega) + 1j * smoothimag(sorted_desired_omega)
            #smooth = si.interp1d(sorted_gf_omega, sorted_func, axis=0, kind='cubic', fill_value='extrapolate')
            #sorted_gf_hat_sm = smooth(sorted_desired_omega)
            sorted_gf_hat_sm = np.zeros((len(sorted_desired_omega), 3), dtype='complex')
            smoothreal = si.UnivariateSpline(sorted_gf_omega, np.real(sorted_func[:,0]), k=4, s=0)
            smoothimag = si.UnivariateSpline(sorted_gf_omega, np.imag(sorted_func[:,0]), k=4, s=0)
            sorted_gf_hat_sm[:,0] = smoothreal(sorted_desired_omega) + 1j * smoothimag(sorted_desired_omega)
            smoothreal = si.UnivariateSpline(sorted_gf_omega, np.real(sorted_func[:,1]), k=4, s=0)
            smoothimag = si.UnivariateSpline(sorted_gf_omega, np.imag(sorted_func[:,1]), k=4, s=0)
            sorted_gf_hat_sm[:,1] = smoothreal(sorted_desired_omega) + 1j * smoothimag(sorted_desired_omega)
            smoothreal = si.UnivariateSpline(sorted_gf_omega, np.real(sorted_func[:,2]), k=4, s=0)
            smoothimag = si.UnivariateSpline(sorted_gf_omega, np.imag(sorted_func[:,2]), k=4, s=0)
            sorted_gf_hat_sm[:,2] = smoothreal(sorted_desired_omega) + 1j * smoothimag(sorted_desired_omega)
            gf_hat_sm = np.concatenate((sorted_gf_hat_sm[-ind:], sorted_gf_hat_sm[:-ind]))
            new_gfs.append(np.fft.ifft(gf_hat_sm, axis=0) / dt)
            if PLOT:
                plt.plot(time, np.fft.ifft(gf_hat_sm[:,1], axis=0) / dt, '.', color=col)
                #plt.plot(time, sint.cumtrapz(np.fft.ifft(gf_hat_sm[:,1], axis=0) / dt, x=time, initial=0), '.', color=col)
                #plt.plot(sorted_gf_omega[:-1], np.abs(smooth(sorted_gf_omega[:-1])[:,1]), color=col)
                #plt.plot(sorted_desired_omega, np.abs(sorted_gf_hat_sm[:,1]), '.', color=col)
    else:
        new_gfs = gfs
    if SAVE:
        for func, lab in zip(new_gfs, components):
            combined = np.concatenate((np.array([time]).transpose(), func), axis=1)
            np.savetxt(save_file+'interpolated_'+lab, combined,
                    header="time, vertical, radial, transverse")
    if PLOT:
        plt.legend()
        plt.show()

    return time, new_gfs

def load_gfs_ES(directory, srctype, time, depths, INTERPOLATE_TIME=False, INTERPOLATE_SPACE=False, 
                SAVE=False, save_file='gf', PLOT=False, REPEATED=0):
    '''
    loads in extended source Green's functions and can interpolate in time/space to get compatible
    array dimensions with desired time/depth array
    can also save the Green's functions to a new file 
    
    NB: if don't want to interpolate each time, can save the resulting interpolated 
        Green's functions and just load them in next time without alteration
    
    original Green's functions must be stored such that the columns correspond below:
            time    vertical    radial   transverse

    --INPUTS--
    directory         : string            : path to folder holding Green's function files
    srctype           : (1)               : source type (0: moment tensor, 1: single force)
    time              : (# time points)   : desired time array
    depths            : (# grid points)   : desired depth array
    INTERPOLATE_TIME  : bool              : if True, interpolate to get values at desired time
    INTERPOLATE_SPACE : bool              : if True, interpolate to get values at desired depths
    SAVE              : bool              : if True, saves Green's functions to save_file
    save_file         : string            : path to where Green's functions will be saved
    --RETURNS--
    gf_time     : (# time points)                       : desired time array
    gfs         : [ (# grid points, # time points, 3) ] : list of final Green's functions (ver, rad, tra)
                                                             if single force, 2 arrays
                                                             if moment tensor, 6 arrays
    '''
    if srctype == 0:
        components = ['Mxx.mat', '2Mxy.mat', '2Mxz.mat', 'Myy.mat', '2Myz.mat', 'Mzz.mat']
    elif srctype == 1:
        components = ['horizontal_force.mat', 'vertical_force.mat']
    colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000', '#E50000']
    
    if not INTERPOLATE_TIME and not INTERPOLATE_SPACE:
        gfs = []
        gf_time = sio.loadmat(directory+'time.mat')['out'][0]
        for com in components:
            gf = sio.loadmat(directory+com)['out']
            gfs.append(gf)
        gc.collect()
        return gf_time, gfs

    gfs = []
    gfs_hat = []
    gf_time = sio.loadmat(directory+'time.mat')['out'][0]
    gf_tt = len(gf_time)
    tt = len(time)
    gf_depths = sio.loadmat(directory+'depths.mat')['out'][:].astype(np.float)*1e3
    gf_hh = len(gf_depths)
    hh = len(depths)
    gf_dt = gf_time[1] - gf_time[0]
    dt = time[1] - time[0]

    for com in components:
        gf = np.zeros((gf_hh, tt, 3))
        gf[:,:gf_tt] = sio.loadmat(directory+com)['out'][:]
        for ii in range(REPEATED):
            gf[-ii] = gf[-REPEATED]
        gfs.append(gf)
        gfs_hat.append(np.fft.fft(gf, axis=1) * gf_dt)
    gc.collect()

    if PLOT:
        #plt.pcolormesh(gf_omega, gf_depths, np.abs(gfs_hat[0][:,:,1]))
        plt.pcolormesh(time, gf_depths, np.real(gfs[0][:,:,1]))
        plt.xlim(0, 100)
        plt.ylim(400, 0)
        plt.ylabel('depth (m)')
        plt.xlabel('time (s)')
        plt.title('GFxx radial BEFORE')
        plt.colorbar()
        plt.show()
        #for func, lab, col in zip(gfs_hat, components, colors):
        #    plt.plot(gf_omega, np.abs(func[:,1]), color=col, label=lab)
        #    #plt.plot(gf_time, func[:,1], color=col, label=lab)
        ##plt.show()

    new_gfs_hat = []
    for func, lab, col in zip(gfs_hat, components, colors):
        smooth = si.interp1d(gf_depths, func, axis=0, kind='linear', fill_value='extrapolate') #bounds_error=False, fill_value=(func[-1], func[0]))
        gf_hat_sm = smooth(depths)
        new_gfs_hat.append(gf_hat_sm)


    #if INTERPOLATE_TIME:
    #    tt = len(time)
    #    desired_omega = np.fft.fftfreq(tt, dt) * (2 * np.pi)
    #    ind = np.argwhere(desired_omega < 0)[0,0]
    #    sorted_desired_omega = np.concatenate((desired_omega[ind:], desired_omega[:ind]))
    #    for func, lab, col in zip(gfs_hat, components, colors):
    #        sorted_func = np.concatenate((func[gf_ind:], func[:gf_ind]))
    #        smooth = si.interp1d(sorted_gf_omega, sorted_func, axis=1, kind='cubic', fill_value='extrapolate')
    #        sorted_gf_hat_sm = smooth(sorted_desired_omega)
    #        #if PLOT:
    #        #    #plt.plot(sorted_gf_omega[:-1], np.abs(smooth(sorted_gf_omega[:-1])[:,1]), color=col)
    #        #    plt.plot(sorted_desired_omega, np.abs(sorted_gf_hat_sm[:,1]), '.', color=col)
    #        gf_hat_sm = np.concatenate((sorted_gf_hat_sm[-ind:], sorted_gf_hat_sm[:-ind]))
    #        new_gfs1_hat.append(gf_hat_sm)
    #        gc.collect()
    #else:
    #    new_gfs1_hat = gfs_hat
    #    gc.collect()
    
    #new_gfs_hat = []
    #if INTERPOLATE_SPACE:
    #    for func, lab, col in zip(new_gfs1_hat, components, colors):
    #        smooth = si.interp1d(gf_depths, func, axis=0, kind='linear', fill_value='extrapolate')
    #        gf_hat_sm = smooth(depths)
    #        new_gfs_hat.append(gf_hat_sm)
    #        gc.collect()
    #else:
    #    new_gfs_hat = new_gfs1_hat
    #    gc.collect()
    
    new_gfs = []
    for gf_h in new_gfs_hat:
        new_gfs.append(np.fft.ifft(gf_h, axis=1) / dt)
    
    if SAVE:
        for func, lab in zip(new_gfs, components):
            sio.savemat(save_file+'interpolated_depths.mat', mdict = {'out' : depths})
            sio.savemat(save_file+'interpolated_time.mat', mdict = {'out' : time})
            sio.savemat(save_file+'interpolated_'+lab, mdict = {'out' : func})

    if PLOT:
        #plt.pcolormesh(desired_omega, depths, np.abs(new_gfs_hat[0][:,:,1]))
        plt.pcolormesh(time, depths, np.real(new_gfs[0][:,:,1]))
        plt.xlim(0, 100)
        plt.ylim(400, 0)
        plt.ylabel('depth (m)')
        plt.xlabel('time (s)')
        plt.title('GFxx radial AFTER')
        plt.colorbar()
        #plt.legend()
        plt.show()

    return time, new_gfs
