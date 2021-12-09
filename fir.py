def moment_general(pressure, depths, time, stationPos, stations, sourceParams, mediumParams, 
                    mt_gf_file, deriv='DIS', coord = 'CYLINDRICAL', SOURCE_FILTER=False, INTERPOLATE=False, 
                    SAVES=False, mt_savefile='greens_functions/'):
    """
    calculates the point source synthetic seismograms from CONDUIT moment contributions at given station 
    positions using loaded Green's functions

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards

        () indicate numpy arrays
        [] indicate lists

    ---INPUTS---
    pressure      : (# sources, # time points) : pressure history along conduit
    depths        : (# sources)                : depths of grid points along conduit
    time          : (# time points)            : time array (assumes equal time-stepping)
    stationPos    : (# stations, 3)            : positions of accelerometer/seismometer stations centered
                                                    around conduit axis
    stations      : [# stations]               : station labels which will be used to load relevant GFs
    sourceParams  : [1, (3)]                   : [conduit area (m^2) OR chamber vol (m^3),
                                                    source position vector]
    mediumParams  : [3]                        : [shear modulus (Pa), lame (Pa), rock density (kg/m^3)]
    mt_gf_file    : string                     : path to directory where MT GF are stored
    deriv         : string                     : seismogram time derivative to return
                                                 (options: 'ACC' acceleration;
                                                           'VEL' velocity;
                                                           'DIS' displacement)
    coord         : string                     : coordinate system for the synthetic seimograms
                                                 (options: 'CARTESIAN' and 'CYLINDRICAL')
    SOURCE_FILTER : bool                       : if True, filters source function before synth-seis calc
    INTERPOLATE   : bool                       : if True, will interpolate loaded GF
    SAVES         : bool                       : if True, will save final GF in savefile directory
    mt_savefile   : string                     : path to directory where final MT GF are saved
    ---RETURNS---
    $$_x, $$_y, $$_z        : (# stations, # time points) : chosen deriv applied to syn seis if CARTESIAN
    OR $$_r, $$_z, $$_tr                                        OR CYLINDRICAL
    moment                  : (# time points)             : moment history
    """
    dt = time[2] - time[1]

    sourceDim, sourcePos = sourceParams
    mu, lame, rho_rock = mediumParams                                               

    # converting position vectors into cylindrical coordinates (to use in SW calc)
    # +z downwards (just for SW calculations)
    stationPos_cyl = cylindrical(stationPos)

    # setting up low-pass filter to eliminate high frequency numerical effects
    nyq_freq = 0.5 / dt  # in Hz
    cutoff_freq = 0.03  # in Hz
    normal_cutoff = cutoff_freq / nyq_freq
    b, a = sg.butter(3, normal_cutoff, btype='low', analog=False)

    shift = 15000
    dmoment_dz = ss.moment_density(pressure, sourceDim, cushion=shift)
    if SOURCE_FILTER:
        filt = sg.lfilter(b, a, dmoment_dz)[:,shift:]
    else:
        filt = dmoment_dz[:,shift:]
    moment = hp.integration_trapezoid(depths, np.array([filt]))
    dmoment_dz = filt
    moment_tensor = ss.moment_tensor_cylindricalSource([lame, mu])
    gc.collect()

    dmoment_rate = np.gradient(dmoment_dz, dt, axis=-1)
    #plt.plot(depths, dmoment_rate[:,0])
    #plt.plot(time, dmoment_rate[0])
    #plt.plot(time, dmoment_rate[-1])
    #plt.show()
    #plt.pcolormesh(time, depths, dmoment_rate)
    #plt.xlim(0, 1500)
    #plt.ylim(1000, 0)
    #plt.ylabel('depth (m)')
    #plt.xlabel('time (s)')
    #plt.title('d(moment rate)/dz')
    #plt.colorbar()
    #plt.show()
    gc.collect()

    TT = np.ma.size(dmoment_dz, axis=1)  # number of time points
    NN = np.ma.size(stationPos, axis=0)  # number of receivers
    HH = np.ma.size(dmoment_dz, axis=0)  # number of sources

    dmom_rate_hat = np.fft.fft(dmoment_rate, axis=1) * dt
    gc.collect()

    #dvel_z = np.zeros((NN, HH, TT), dtype='complex')
    #dvel_r = np.zeros((NN, HH, TT), dtype='complex')
    #dvel_tr = np.zeros((NN, HH, TT), dtype='complex')

    repeated_depth = [10, 15, 16, 16]
    #repeated_depth = [16, 16]

    vel_z = np.zeros((NN, TT), dtype='complex')
    vel_r = np.zeros((NN, TT), dtype='complex')
    vel_tr = np.zeros((NN, TT), dtype='complex')
    for stat, ii in zip(stations, np.arange(NN)):
        print(stat)
        gf_time, gfs = gf.load_gfs_ES(mt_gf_file+stat+'/', 0, time, depths, INTERPOLATE_TIME=INTERPOLATE, INTERPOLATE_SPACE=INTERPOLATE,
                                    SAVE=SAVES, save_file=mt_savefile+stat+'/', PLOT=False, REPEATED=repeated_depth[ii])
        #print(np.max(abs(gfs[0][:,:,2])))
        print('loaded gfs...')
        gfs_hat = []
        for gg in gfs:
            gf_hat = np.fft.fft(gg, axis=1) * dt
            gfs_hat.append(gf_hat)
            gc.collect()
        print('fourier transformed gfs...')
        dvel_z = np.fft.ifft(dmom_rate_hat * moment_tensor[0,0] * gfs_hat[0][:,:,0], axis=-1) / dt
        dvel_z += np.fft.ifft(dmom_rate_hat * moment_tensor[0,1] * gfs_hat[1][:,:,0], axis=-1) / dt
        dvel_z += np.fft.ifft(dmom_rate_hat * moment_tensor[0,2] * gfs_hat[2][:,:,0], axis=-1) / dt
        dvel_z += np.fft.ifft(dmom_rate_hat * moment_tensor[1,1] * gfs_hat[3][:,:,0], axis=-1) / dt
        dvel_z += np.fft.ifft(dmom_rate_hat * moment_tensor[1,2] * gfs_hat[4][:,:,0], axis=-1) / dt
        dvel_z += np.fft.ifft(dmom_rate_hat * moment_tensor[2,2] * gfs_hat[5][:,:,0], axis=-1) / dt
        plt.pcolormesh(time, depths, np.real(dvel_z))
        plt.xlim(0, 150)
        plt.ylim(400, 0)
        plt.ylabel('depth (m)')
        plt.xlabel('time (s)')
        plt.title('d(vel_z)/dz')
        plt.colorbar()
        plt.show()
        vel_z[ii] = hp.integration_trapezoid(depths, np.array([dvel_z]))[0]
        gc.collect()

        dvel_r = np.fft.ifft(dmom_rate_hat * moment_tensor[0,0] * gfs_hat[0][:,:,1], axis=-1) / dt
        dvel_r += np.fft.ifft(dmom_rate_hat * moment_tensor[0,1] * gfs_hat[1][:,:,1], axis=-1) / dt
        dvel_r += np.fft.ifft(dmom_rate_hat * moment_tensor[0,2] * gfs_hat[2][:,:,1], axis=-1) / dt
        dvel_r += np.fft.ifft(dmom_rate_hat * moment_tensor[1,1] * gfs_hat[3][:,:,1], axis=-1) / dt
        dvel_r += np.fft.ifft(dmom_rate_hat * moment_tensor[1,2] * gfs_hat[4][:,:,1], axis=-1) / dt
        dvel_r += np.fft.ifft(dmom_rate_hat * moment_tensor[2,2] * gfs_hat[5][:,:,1], axis=-1) / dt
        plt.pcolormesh(time, depths, np.real(dvel_r))
        plt.xlim(0, 150)
        plt.ylim(400, 0)
        plt.ylabel('depth (m)')
        plt.xlabel('time (s)')
        plt.title('d(vel_r)/dz')
        plt.colorbar()
        plt.show()
        vel_r[ii] = hp.integration_trapezoid(depths, np.array([dvel_r]))[0]
        gc.collect()

        dvel_tr = np.fft.ifft(dmom_rate_hat * moment_tensor[0,0] * gfs_hat[0][:,:,2], axis=-1) / dt
        dvel_tr += np.fft.ifft(dmom_rate_hat * moment_tensor[0,1] * gfs_hat[1][:,:,2], axis=-1) / dt
        dvel_tr += np.fft.ifft(dmom_rate_hat * moment_tensor[0,2] * gfs_hat[2][:,:,2], axis=-1) / dt
        dvel_tr += np.fft.ifft(dmom_rate_hat * moment_tensor[1,1] * gfs_hat[3][:,:,2], axis=-1) / dt
        dvel_tr += np.fft.ifft(dmom_rate_hat * moment_tensor[1,2] * gfs_hat[4][:,:,2], axis=-1) / dt
        dvel_tr += np.fft.ifft(dmom_rate_hat * moment_tensor[2,2] * gfs_hat[5][:,:,2], axis=-1) / dt
        plt.pcolormesh(time, depths, np.real(dvel_tr))
        plt.xlim(0, 150)
        plt.ylim(300, 0)
        plt.ylabel('depth (m)')
        plt.xlabel('time (s)')
        plt.title('d(vel_tr)/dz')
        plt.colorbar()
        plt.show()
        vel_tr[ii] = hp.integration_trapezoid(depths, np.array([dvel_tr]))[0]
        gc.collect()

        print('finished convolution')
        print('---------------------------------------------')

    #vel_z = hp.integration_trapezoid(depths, dvel_z)
    #vel_r = hp.integration_trapezoid(depths, dvel_r)
    #vel_tr = hp.integration_trapezoid(depths, dvel_tr)
    gc.collect()
    print('finished integration')

    vel_z = np.real(vel_z)
    vel_r = np.real(vel_r)
    vel_tr = np.real(vel_tr)

    if coord == 'CARTESIAN':
        vel_x, vel_y = cartesian(vel_r, stationPos_cyl)
        gc.collect()
        if deriv == 'ACC':
            acc_x = np.gradient(vel_x, dt, axis=1)
            acc_y = np.gradient(vel_y, dt, axis=1)
            acc_z = np.gradient(vel_z, dt, axis=1)
            gc.collect()
            return acc_x, acc_y, acc_z, moment[0]
        elif deriv == 'DIS':
            dis_x = sint.cumtrapz(vel_x, x=time, initial=0)
            dis_y = sint.cumtrapz(vel_y, x=time, initial=0)
            dis_z = sint.cumtrapz(vel_z, x=time, initial=0)
            gc.collect()
            return dis_x, dis_y, dis_z, moment[0]
        else:
            return vel_x, vel_y, vel_z, moment[0]
    else:
        if deriv == 'ACC':
            acc_r = np.gradient(vel_r, dt, axis=1)
            acc_z = np.gradient(vel_z, dt, axis=1)
            acc_tr = np.gradient(vel_tr, dt, axis=1)
            gc.collect()
            return acc_r, acc_z, acc_tr, moment[0]
        elif deriv == 'DIS':
            dis_r = sint.cumtrapz(vel_r, x=time, initial=0)
            dis_z = sint.cumtrapz(vel_z, x=time, initial=0)
            dis_tr = sint.cumtrapz(vel_tr, x=time, initial=0)
            gc.collect()
            return dis_r, dis_z, dis_tr, moment[0]
        else:
            return vel_r, vel_z, vel_tr, moment[0]


