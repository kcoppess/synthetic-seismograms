#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gc
import os
import io
from zipfile import ZipFile
import scipy.signal as sg
import scipy.interpolate as si
import scipy.integrate as sint

import helpers as hp
import source_setup as ss
import bodywaves_functions as bw
import surfacewaves_functions as sw
import load_gfs as gf


def cylindrical(original):
    """
    transforms from Cartesian to cylindrical (r, phi, z)

    NB: +z downwards

    ---INPUTS---
    original : (# stations, 3)
    ---RETURNS---
    new      : (# stations, 3)
    """
    nn = original.shape[0]

    new = np.zeros((nn, 3))
    phis = np.zeros(nn)
    for ii in range(nn):
        if original[ii][0] < 0:
            phi = np.arctan(original[ii][1] / original[ii][0]) + np.pi
        elif original[ii][0] == 0:
            if original[ii][1] > 0:
                phi = np.pi / 2
            else:
                phi = 3 * np.pi / 2
        else:
            phi = np.arctan(original[ii][1] / original[ii][0])
        new[ii] = np.array([np.linalg.norm(original[ii][:2]), phi, -original[ii][2]])
    return new


def cartesian(radial, pos_cyl):
    """
    convert radial displacement into cartesian x, y
    """
    nn = radial.shape[0]
    hh = radial.shape[1]

    x = np.zeros(radial.shape, dtype='complex')
    y = np.zeros(radial.shape, dtype='complex')
    for ii in range(nn):
        for jj in range(hh):
            x[ii, jj] = radial[ii, jj] * np.cos(pos_cyl[jj, 1])
            y[ii, jj] = radial[ii, jj] * np.sin(pos_cyl[jj, 1])
    return x, y


def moment_general(SOURCE_TYPE, pressure, depths, time, stationPos, stations, sourceParams, mediumParams, 
                    mt_gf_file, deriv='DIS', coord = 'CYLINDRICAL', SOURCE_FILTER=False, INTERPOLATE=False, 
                    SAVES=False, mt_savefile='greens_functions/'):
    """
    calculates the point source synthetic seismograms from moment contributions at given station positions
    using loaded Green's functions

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards

        () indicate numpy arrays
        [] indicate lists

    ---INPUTS---
    SOURCE_TYPE   : string                     : either 'CONDUIT' or 'CHAMBER'
    pressure      : (# time points)            : CHAMBER -> chamber pressure history
               OR : (# sources, # time points) : CONDUIT -> pressure history along conduit
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

    if SOURCE_TYPE == 'CONDUIT':
        dmoment_dz = ss.moment_density(pressure, sourceDim, cushion=shift)
        if SOURCE_FILTER:
            filt = sg.lfilter(b, a, dmoment_dz)
        else:
            filt = dmoment_dz
        moment = hp.integration_trapezoid(height, np.array([filt]))
        moment_tensor = ss.moment_tensor_cylindricalSource([lame, mu])
    elif SOURCE_TYPE == 'CHAMBER':
        moment_unfil = ss.moment_density(np.array([pressure]), sourceDim, cushion=shift)[0]
        if SOURCE_FILTER:
            moment = [sg.lfilter(b, a, moment_unfil)]
        else:
            moment = [moment_unfil]
        moment_tensor = np.eye(3) * ((lame + 2 * mu) / mu)
    gc.collect()

    moment_rate = np.gradient(moment, dt, axis=-1)
    gc.collect()

    TT = np.ma.size(moment, axis=1)  # number of time points
    NN = np.ma.size(stationPos, axis=0)  # number of receivers

    mom_rate_hat = np.fft.fft(moment_rate, axis=1) * dt

    vel_z = np.zeros((1, NN, TT), dtype='complex')
    vel_r = np.zeros((1, NN, TT), dtype='complex')
    vel_tr = np.zeros((1, NN, TT), dtype='complex')

    for stat, ii in zip(stations, np.arange(NN)):
        gf_time, gfs = gf.load_gfs(mt_gf_file+stat+'/', 0, time, INTERPOLATE_TIME=INTERPOLATE, SAVE=SAVES, 
                                    save_file=mt_savefile, PLOT=False)
        gfs_hat = []
        for gg in gfs:
            gf_hat = np.fft.fft(gg, axis=0) * dt
            gfs_hat.append(gf_hat)
        vel_z[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,0] * gfs_hat[0][:,0], axis=-1) / dt
        vel_z[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,1] * gfs_hat[1][:,0], axis=-1) / dt
        vel_z[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,2] * gfs_hat[2][:,0], axis=-1) / dt
        vel_z[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[1,1] * gfs_hat[3][:,0], axis=-1) / dt
        vel_z[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[1,2] * gfs_hat[4][:,0], axis=-1) / dt
        vel_z[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[2,2] * gfs_hat[5][:,0], axis=-1) / dt
        gc.collect()

        vel_r[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,0] * gfs_hat[0][:,1], axis=-1) / dt
        vel_r[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,1] * gfs_hat[1][:,1], axis=-1) / dt
        vel_r[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,2] * gfs_hat[2][:,1], axis=-1) / dt
        vel_r[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[1,1] * gfs_hat[3][:,1], axis=-1) / dt
        vel_r[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[1,2] * gfs_hat[4][:,1], axis=-1) / dt
        vel_r[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[2,2] * gfs_hat[5][:,1], axis=-1) / dt
        gc.collect()

        vel_tr[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,0] * gfs_hat[0][:,2], axis=-1) / dt
        vel_tr[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,1] * gfs_hat[1][:,2], axis=-1) / dt
        vel_tr[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[0,2] * gfs_hat[2][:,2], axis=-1) / dt
        vel_tr[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[1,1] * gfs_hat[3][:,2], axis=-1) / dt
        vel_tr[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[1,2] * gfs_hat[4][:,2], axis=-1) / dt
        vel_tr[0, ii] += np.fft.ifft(mom_rate_hat[0] * moment_tensor[2,2] * gfs_hat[5][:,2], axis=-1) / dt

    vel_z = np.real(vel_z)
    vel_r = np.real(vel_r)
    vel_tr = np.real(vel_tr)

    if coord == 'CARTESIAN':
        vel_x, vel_y = cartesian(vel_r, stationPos_cyl)
        if deriv == 'ACC':
            acc_x = np.gradient(vel_x[0], dt, axis=1)
            acc_y = np.gradient(vel_y[0], dt, axis=1)
            acc_z = np.gradient(vel_z[0], dt, axis=1)
            return acc_x, acc_y, acc_z, moment[0]
        elif deriv == 'DIS':
            dis_x = sint.cumtrapz(vel_x[0], x=time, initial=0)
            dis_y = sint.cumtrapz(vel_y[0], x=time, initial=0)
            dis_z = sint.cumtrapz(vel_z[0], x=time, initial=0)
            return dis_x, dis_y, dis_z, moment[0]
        else:
            return vel_x, vel_y, vel_z, moment[0]
    else:
        if deriv == 'ACC':
            acc_r = np.gradient(vel_r[0], dt, axis=1)
            acc_z = np.gradient(vel_z[0], dt, axis=1)
            acc_tr = np.gradient(vel_tr[0], dt, axis=1)
            return acc_r, acc_z, acc_tr, moment[0]
        elif deriv == 'DIS':
            dis_r = sint.cumtrapz(vel_r[0], x=time, initial=0)
            dis_z = sint.cumtrapz(vel_z[0], x=time, initial=0)
            dis_tr = sint.cumtrapz(vel_tr[0], x=time, initial=0)
            return dis_r, dis_z, dis_tr, moment[0]
        else:
            return vel_r, vel_z, vel_tr, moment[0]


def force_general(SOURCE_TYPE, force, depths, time, stationPos, stations, sourceParams, mediumParams, 
                    sf_gf_file, deriv='DIS', coord = 'CYLINDRICAL', SOURCE_FILTER=False, INTERPOLATE=False, 
                    SAVES=False, sf_savefile='greens_functions/'):
    """
    calculates the point source synthetic seismograms from single force contributions at given station positions
    using loaded Green's functions

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards

        () indicate numpy arrays
        [] indicate lists

    ---INPUTS---
    SOURCE_TYPE   : string                     : either 'CONDUIT' or 'CHAMBER'
    force         : (# time points)            : CHAMBER -> chamber-conduit momentum exchange history
               OR : (# sources, # time points) : CONDUIT -> shear traction history along conduit
    depths        : (# sources)                : depths of grid points along conduit
    time          : (# time points)            : time array (assumes equal time-stepping)
    stationPos    : (# stations, 3)            : positions of accelerometer/seismometer stations centered
                                                    around conduit axis
    stations      : [# stations]               : station labels which will be used to load relevant GFs
    sourceParams  : [1, (3)]                   : [conduit area (m^2) OR chamber vol (m^3),
                                                    source position vector]
    mediumParams  : [3]                        : [shear modulus (Pa), lame (Pa), rock density (kg/m^3)]
    sf_gf_file    : string                     : path to directory where SF GF are stored
    deriv         : string                     : seismogram time derivative to return
                                                 (options: 'ACC' acceleration;
                                                           'VEL' velocity;
                                                           'DIS' displacement)
    coord         : string                     : coordinate system for the synthetic seimograms
                                                 (options: 'CARTESIAN' and 'CYLINDRICAL')
    SOURCE_FILTER : bool                       : if True, filters source function before synth-seis calc
    INTERPOLATE   : bool                       : if True, will interpolate loaded GF
    SAVES         : bool                       : if True, will save final GF in savefile directory
    sf_savefile   : string                     : path to directory where final SF GF are saved
    ---RETURNS---
    $$_x, $$_y, $$_z        : (# stations, # time points) : chosen deriv applied to syn seis if CARTESIAN
    OR $$_r, $$_z, $$_tr                                        OR CYLINDRICAL
    force                   : (# time points)             : force history
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

    if SOURCE_TYPE == 'CONDUIT':
        dforce_dz = ss.moment_density(force, sourceDim, cushion=shift)
        if SOURCE_FILTER:
            filt = sg.lfilter(b, a, dforce_dz)
        else:
            filt = dforce_dz
        force = hp.integration_trapezoid(height, np.array([filt]))
    elif SOURCE_TYPE == 'CHAMBER':
        force_unfil = ss.moment_density(np.array([force]), sourceDim, cushion=shift)[0]
        if SOURCE_FILTER:
            force = np.array([sg.lfilter(b, a, force_unfil)])
        else:
            force = np.array([force_unfil])
    gc.collect()
    
    force_rate = np.gradient(force, dt, axis=-1)
    gc.collect()

    TT = np.ma.size(force, axis=1)  # number of time points
    NN = np.ma.size(stationPos, axis=0)  # number of receivers

    for_rate_hat = np.fft.fft(force_rate, axis=1) * dt

    vel_z = np.zeros((1, NN, TT), dtype='complex')
    vel_r = np.zeros((1, NN, TT), dtype='complex')
    vel_tr = np.zeros((1, NN, TT), dtype='complex')

    for stat, ii in zip(stations, np.arange(NN)):
        gf_time, gfs = gf.load_gfs(mt_gf_file+stat+'/', 1, time, INTERPOLATE_TIME=INTERPOLATE, SAVE=SAVES, save_file=sf_savefile, PLOT=False)
        gfs_hat = []
        for gg in gfs:
            gf_hat = np.fft.fft(gg, axis=0) * dt
            gfs_hat.append(gf_hat)
        vel_z[0, ii] += np.fft.ifft(for_rate_hat[0] * gfs_hat[1][:,0], axis=-1) / dt
        vel_r[0, ii] += np.fft.ifft(for_rate_hat[0] * gfs_hat[1][:,1], axis=-1) / dt
        vel_tr[0, ii] += np.fft.ifft(for_rate_hat[0] * gfs_hat[1][:,2], axis=-1) / dt
        gc.collect()

    vel_z = np.real(vel_z)
    vel_r = np.real(vel_r)
    vel_tr = np.real(vel_tr)

    if coord == 'CARTESIAN':
        vel_x, vel_y = cartesian(vel_r, stationPos_cyl)
        if deriv == 'ACC':
            acc_x = np.gradient(vel_x[0], dt, axis=1)
            acc_y = np.gradient(vel_y[0], dt, axis=1)
            acc_z = np.gradient(vel_z[0], dt, axis=1)
            return acc_x, acc_y, acc_z, force[0]
        elif deriv == 'DIS':
            dis_x = sint.cumtrapz(vel_x[0], x=time, initial=0)
            dis_y = sint.cumtrapz(vel_y[0], x=time, initial=0)
            dis_z = sint.cumtrapz(vel_z[0], x=time, initial=0)
            return dis_x, dis_y, dis_z, force[0]
        else:
            return vel_x, vel_y, vel_z, force[0]
    else:
        if deriv == 'ACC':
            acc_r = np.gradient(vel_r[0], dt, axis=1)
            acc_z = np.gradient(vel_z[0], dt, axis=1)
            acc_tr = np.gradient(vel_tr[0], dt, axis=1)
            return acc_r, acc_z, acc_tr, force[0]
        elif deriv == 'DIS':
            dis_r = sint.cumtrapz(vel_r[0], x=time, initial=0)
            dis_z = sint.cumtrapz(vel_z[0], x=time, initial=0)
            dis_tr = sint.cumtrapz(vel_tr[0], x=time, initial=0)
            return dis_r, dis_z, dis_tr, force[0]
        else:
            return vel_r, vel_z, vel_tr, force[0]


def moment_mixed_analytical(SOURCE_TYPE, pressure, height, dt, stationPos, sourceParams, mediumParams, wave_terms, ps_waves,
                     WAVE='BOTH', deriv='DIS', SOURCE_FILTER=False):
    """
    calculates the point source synthetic seismograms from moment contributions at given station positions

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards

        () indicate numpy arrays
        [] indicate lists

    ---INPUTS---
    pressure      : (# time points)            : CHAMBER -> chamber pressure history
               OR : (# sources, # time points) : CONDUIT -> pressure history along conduit
    height        : (# sources)                : depths of grid points along conduit
                                                    NB: different sign convention but just used for integration
    dt            : (dt)                       : time step size (assumes equal time-stepping)
    stationPos    : (# stations, 3)            : positions of accelerometer/seismometer stations centered
                                                    around conduit axis
    sourceParams  : [1, (3)]                   : [conduit area (m^2) OR chamber vol (m^3),
                                                    source position vector]
    mediumParams  : [2]                        : [shear modulus (Pa), rock density (kg/m^3)]
                                                 (assumes Poisson ratio = 1/4)
    WAVE          : string                     : 'BOTH' calculates both body and surface waves (default)
                                                 'BODY' just calculate body waves
                                                 'SURF' just calculate surface waves
    deriv         : string                     : seismogram time derivative to return
                                                 (options: 'ACC' acceleration;
                                                           'VEL' velocity;
                                                           'DIS' displacement)
    SOURCE_FILTER : bool                       : if True, filters source function before synth-seis calc
    ---RETURNS---
    seismo_x, seismo_y, seismo_z : (# stations, # time points) : chosen deriv applied to synthetic seismograms
    """
    sourceDim, sourcePos = sourceParams
    mu, rho_rock = mediumParams
    lame = mu  # poisson ratio = 1/4

    # converting position vectors into cylindrical coordinates (to use in SW calc)
    # +z downwards (just for SW calculations)
    stationPos_cyl = cylindrical(stationPos)

    # storing coordinates for separation vectors between point-source and seismometers
    # separation: (# receivers, # sources)
    # gamma: (# receivers, # sources, 3)
    separation, gamma = hp.separation_distances_vectors(stationPos, [sourcePos])
    gc.collect()

    # phase shift so as to eliminate some edge effects in from fourier transformation
    # number of time steps
    shift = 15000

    # setting up low-pass filter to eliminate high frequency numerical effects
    nyq_freq = 0.5 / dt  # in Hz
    cutoff_freq = 0.03  # in Hz
    normal_cutoff = cutoff_freq / nyq_freq
    b, a = sg.butter(3, normal_cutoff, btype='low', analog=False)

    if SOURCE_TYPE == 'CONDUIT':
        dmoment_dz = ss.moment_density(pressure, sourceDim, cushion=shift)
        if SOURCE_FILTER:
            filt = sg.lfilter(b, a, dmoment_dz)
        else:
            filt = dmoment_dz
        moment = hp.integration_trapezoid(height, np.array([filt]))
        moment_tensor = ss.moment_tensor_cylindricalSource([lame, mu])
    elif SOURCE_TYPE == 'CHAMBER':
        moment_unfil = ss.moment_density(np.array([pressure]), sourceDim, cushion=shift)[0]
        if SOURCE_FILTER:
            moment = [sg.lfilter(b, a, moment_unfil)]
        else:
            moment = [moment_unfil]
        if mu == 0:
            moment_tensor = np.eye(3)  # * ((lame + 2*mu) / mu)
        else:
            moment_tensor = np.eye(3) * ((lame + 2 * mu) / mu)
    gc.collect()

    seismo_x = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(moment, axis=1)), dtype='complex')
    seismo_y = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(moment, axis=1)), dtype='complex')
    seismo_z = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(moment, axis=1)), dtype='complex')

    if WAVE == 'BODY' or WAVE == 'BOTH':
        body_x, body_y, body_z = bw.displacement_moment(moment, moment_tensor, separation,
                                                        gamma, dt, [rho_rock, lame, mu], terms=wave_terms,
                                                        ps_tuner=ps_waves)
        seismo_x += body_x
        gc.collect()
        seismo_y += body_y
        gc.collect()
        seismo_z += body_z
        gc.collect()
    if WAVE == 'SURF' or WAVE == 'BOTH':
        surf_r, surf_z = sw.rayleigh_displacement_moment(moment, moment_tensor, stationPos_cyl,
                                                         np.array([-sourcePos[2]]), dt, [rho_rock, lame, mu])
        seismo_z += surf_z
        gc.collect()
        surf_x, surf_y = cartesian(surf_r, stationPos_cyl)
        gc.collect()
        seismo_x += surf_x
        gc.collect()
        seismo_y += surf_y
        gc.collect()

    if deriv == 'ACC':
        return np.gradient(np.gradient(seismo_x[:, 0, shift:], dt, axis=1), dt, axis=1), np.gradient(
            np.gradient(seismo_y[:, 0, shift:], dt, axis=1), dt, axis=1), np.gradient(
            np.gradient(seismo_z[:, 0, shift:], dt, axis=1), dt, axis=1), moment[0][shift:]
    elif deriv == 'VEL':
        return np.gradient(seismo_x[:, 0, shift:], dt, axis=1), np.gradient(seismo_y[:, 0, shift:], dt,
                                                                            axis=1), np.gradient(seismo_z[:, 0, shift:],
                                                                                                 dt, axis=1), moment[0][
                                                                                                              shift:]
    else:
        return seismo_x[:, 0, shift:], seismo_y[:, 0, shift:], seismo_z[:, 0, shift:], moment[0][shift:]


def force_mixed_analytical(SOURCE_TYPE, force, height, dt, stationPos, sourceParams, mediumParams, wave_terms, ps_waves,
                    WAVE='BOTH', deriv='DIS', SOURCE_FILTER=False):
    """
    calculates the point source synthetic seismograms from force contributions at given station positions

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards

        () indicate numpy arrays
        [] indicate lists

    ---INPUTS---
    force         : (# time points)            : CHAMBER -> chamber pressure history
               OR : (# sources, # time points) : CONDUIT -> shear force history along conduit
    height        : (# sources)                : depths of grid points along conduit
                                                    NB: different sign convention but just used for integration
    dt            : (dt)                       : time step size (assumes equal time-stepping)
    stationPos    : (# stations, 3)            : positions of accelerometer/seismometer stations centered
                                                    around conduit axis
    sourceParams  : [1, (3)]                   : [conduit area (m^2), source position vector]
    mediumParams  : [2]                        : [shear modulus (Pa), rock density (kg/m^3)]
                                                 (assumes Poisson ratio = 1/4)
    WAVE          : string                     : 'BOTH' calculates both body and surface waves (default)
                                                 'BODY' just calculate body waves
                                                 'SURF' just calculate surface waves
    deriv         : string                     : seismogram time derivative to return
                                                 (options: 'ACC' acceleration;
                                                           'VEL' velocity;
                                                           'DIS' displacement)
    SOURCE_FILTER : bool                       : if True, filters source function before synth-seis calc
    ---RETURNS---
    seismo_x, seismo_y, seismo_z : (# stations, # time points) : chosen deriv applied to synthetic seismograms
    """
    sourceDim, sourcePos = sourceParams
    mu, rho_rock = mediumParams
    lame = mu  # poisson ratio = 1/4

    # converting position vectors into cylindrical coordinates (to use in SW calc)
    # +z downwards (just for SW calculations)
    stationPos_cyl = cylindrical(stationPos)

    # storing coordinates for separation vectors between point-source and seismometers
    # separation: (# receivers, # sources)
    # gamma: (# receivers, # sources, 3)
    separation, gamma = hp.separation_distances_vectors(stationPos, [sourcePos])
    gc.collect()

    # phase shift so as to eliminate some edge effects in from fourier transformation
    # number of time steps
    shift = 15000

    # setting up low-pass filter to eliminate high frequency numerical effects
    nyq_freq = 0.5 / dt  # in Hz
    cutoff_freq = 0.03  # in Hz
    normal_cutoff = cutoff_freq / nyq_freq
    b, a = sg.butter(3, normal_cutoff, btype='low', analog=False)

    if SOURCE_TYPE == 'CONDUIT':
        dforce_dz = ss.moment_density(force, sourceDim, cushion=shift)
        if SOURCE_FILTER:
            filt = sg.lfilter(b, a, dforce_dz)
        else:
            filt = dforce_dz
        force = hp.integration_trapezoid(height, np.array([filt]))
    elif SOURCE_TYPE == 'CHAMBER':
        force_unfil = ss.moment_density(np.array([force]), sourceDim, cushion=shift)[0]
        if SOURCE_FILTER:
            force = np.array([sg.lfilter(b, a, force_unfil)])
        else:
            force = np.array([force_unfil])
    gc.collect()

    seismo_x = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(force, axis=1)), dtype='complex')
    seismo_y = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(force, axis=1)), dtype='complex')
    seismo_z = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(force, axis=1)), dtype='complex')

    if WAVE == 'BODY' or WAVE == 'BOTH':
        body_x, body_y, body_z = bw.displacement_force(force, separation, gamma, dt, [rho_rock, lame, mu],
                                                       terms=wave_terms, ps_tuner=ps_waves)
        seismo_x += body_x
        gc.collect()
        seismo_y += body_y
        gc.collect()
        seismo_z += body_z
        gc.collect()
    if WAVE == 'SURF' or WAVE == 'BOTH':
        surf_r, surf_z = sw.rayleigh_displacement_force(-force, stationPos_cyl, np.array([-sourcePos[2]]), dt,
                                                        [rho_rock, lame, mu])
        seismo_z += surf_z
        gc.collect()
        surf_x, surf_y = cartesian(surf_r, stationPos_cyl)
        gc.collect()
        seismo_x += surf_x
        gc.collect()
        seismo_y += surf_y
        gc.collect()

    if deriv == 'ACC':
        return np.gradient(np.gradient(seismo_x[:, 0, shift:], dt, axis=1), dt, axis=1), np.gradient(
            np.gradient(seismo_y[:, 0, shift:], dt, axis=1), dt, axis=1), np.gradient(
            np.gradient(seismo_z[:, 0, shift:], dt, axis=1), dt, axis=1), force[0][shift:]
    elif deriv == 'VEL':
        return np.gradient(seismo_x[:, 0, shift:], dt, axis=1), np.gradient(seismo_y[:, 0, shift:], dt,
                                                                            axis=1), np.gradient(seismo_z[:, 0, shift:],
                                                                                                 dt, axis=1), force[0][
                                                                                                              shift:]
    else:
        return seismo_x[:, 0, shift:], seismo_y[:, 0, shift:], seismo_z[:, 0, shift:], force[0][shift:]

