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

import helpers as hp
import source_setup as ss
import bodywaves_functions as bw
import surfacewaves_functions as sw

def cylindrical(original):
    '''
    transforms from Cartesian to cylindrical (r, phi, z)
    
    NB: +z downwards
    
    ---INPUTS---
    original : (# stations, 3)
    ---RETURNS---
    new      : (# stations, 3)
    '''
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
    '''
    convert radial displacement into cartesian x, y
    '''
    nn = radial.shape[0]
    
    x = np.zeros(radial.shape)
    y = np.zeros(radial.shape)
    for ii in range(nn):
        x[ii] = radial[ii] * np.cos(pos_cyl[ii,1])
        y[ii] = radial[ii] * np.sin(pos_cyl[ii,1])
    return x, y


def moment_ZIP_load(ZIPFILE, SOURCE_TYPE, TOTAL_TIME):
    '''
    loads pressure time series and applies signal processing (smoothing and constant
    time-stepping)

    ---INPUTS---
    ZIPFILE    : string : full path to data zip file
    SOURCE_TYPE: string : 'CONDUIT' -> load conduit pressure
                          'CHAMBER' -> load chamber pressure
    TOTAL_TIME : 1      : total simulation time
    ---RETURNS---
    p     : (# sources, # time points) : CONDUIT
            (# time points)            : CHAMBER
    time  : (# time points)
    height: (# sources)                : NB positive z is up and z=0 is conduit bottom
    '''
    directory = ZipFile(ZIPFILE, mode='r')

    if SOURCE_TYPE == 'CONDUIT':
        p1 = np.loadtxt(io.BytesIO(directory.read('pressure.txt')), delimiter=',')
        p1 = np.real(p1)
    elif SOURCE_TYPE == 'CHAMBER':
        p1 = np.loadtxt(io.BytesIO(directory.read('chamber_pressure.txt')), delimiter=',')
        p1 = np.real(p1)
    
    time1 = np.loadtxt(io.BytesIO(directory.read('time.txt')), delimiter=',')
    # in conduit flow code, this takes up to be positive z and
    # bottom of conduit is at z = 0
    height = np.loadtxt(io.BytesIO(directory.read('height.txt')), delimiter=',')
    
    directory.close()

    '''signal processing to smooth out numerical effects (e.g. from downsampling)'''
    dt = time1[2] - time1[1]
    length = int(TOTAL_TIME/dt)
    time = np.arange(length)*dt
    
    time_smooth = si.interp1d(np.arange(len(time1))[::2], time1[::2], kind='cubic')
    times = time_smooth(np.arange(len(time1))[:-2])
    
    if SOURCE_TYPE == 'CONDUIT':
        smooth = si.interp1d(times, p1[:,:-2], kind='cubic', axis=1)
        p = smooth(time)
    elif SOURCE_TYPE == 'CHAMBER':
        smooth = si.interp1d(times, p1[:-2], kind='cubic')
        p = smooth(time)
    
    gc.collect()

    return p, time, height


def force_ZIP_load(ZIPFILE, SOURCE_TYPE, TOTAL_TIME):
    '''
    loads force time series and applies signal processing (smoothing and constant
    time-stepping)

    ---INPUTS---
    ZIPFILE    : string : full path to data zip file
    SOURCE_TYPE: string : 'CONDUIT' -> load conduit shear pressure
                          'CHAMBER' -> load chamber pressure
    TOTAL_TIME : 1      : total simulation time
    ---RETURNS---
    f     : (# sources, # time points) : CONDUIT
            (# time points)            : CHAMBER
    time  : (# time points)
    height: (# sources)                : NB positive z is up and z=0 is conduit bottom
    '''
    directory = ZipFile(ZIPFILE, mode='r')

    if SOURCE_TYPE == 'CONDUIT':
        f1 = np.loadtxt(io.BytesIO(directory.read('wall_trac.txt')), delimiter=',')
        f1 = -np.real(f1)
    elif SOURCE_TYPE == 'CHAMBER':
        f1 = np.loadtxt(io.BytesIO(directory.read('chamber_pressure.txt')), delimiter=',')
        f1 = -np.real(f1)
    
    time1 = np.loadtxt(io.BytesIO(directory.read('time.txt')), delimiter=',')
    # in conduit flow code, this takes up to be positive z and
    # bottom of conduit is at z = 0
    height = np.loadtxt(io.BytesIO(directory.read('height.txt')), delimiter=',')
    
    directory.close()

    '''signal processing to smooth out numerical effects (e.g. from downsampling)'''
    dt = time1[2] - time1[1]
    length = int(TOTAL_TIME/dt)
    time = np.arange(length)*dt
    
    time_smooth = si.interp1d(np.arange(len(time1))[::2], time1[::2], kind='cubic')
    times = time_smooth(np.arange(len(time1))[:-2])
    
    if SOURCE_TYPE == 'CONDUIT':
        smooth = si.interp1d(times, f1[:,:-2], kind='cubic', axis=1)
        f = smooth(time)
    elif SOURCE_TYPE == 'CHAMBER':
        smooth = si.interp1d(times, f1[:-2], kind='cubic')
        f = smooth(time)
    
    gc.collect()

    return f, time, height


def moment_synthetic(SOURCE_TYPE, pressure, height, dt, stationPos, sourceParams, mediumParams, 
                        WAVE='BOTH', deriv='DIS', SOURCE_FILTER=False):
    '''
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
    '''
    sourceDim, sourcePos = sourceParams
    mu, rho_rock = mediumParams
    lame = mu # poisson ratio = 1/4
    
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
    nyq_freq = 0.5 / dt # in Hz
    cutoff_freq = 0.03 # in Hz
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
        moment_tensor = ((lame + 2*mu) / mu) * np.eye(3)
    gc.collect()
    
    seismo_x = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(moment, axis=1)), dtype='complex')
    seismo_y = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(moment, axis=1)), dtype='complex')
    seismo_z = np.zeros((1, np.ma.size(stationPos, axis=0), np.ma.size(moment, axis=1)), dtype='complex')
    
    if WAVE == 'BODY' or WAVE == 'BOTH':
        body_x, body_y, body_z = bw.displacement_moment(moment, moment_tensor, separation,
                                                          gamma, dt, [rho_rock, lame, mu])
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
        return np.gradient(np.gradient(seismo_x[:,0,shift:], dt, axis=1), dt, axis=1), np.gradient(np.gradient(seismo_y[:,0,shift:], dt, axis=1),     dt, axis=1), np.gradient(np.gradient(seismo_z[:,0,shift:], dt, axis=1), dt, axis=1), moment[0][shift:]
    elif deriv == 'VEL':
        return np.gradient(seismo_x[:,0,shift:], dt, axis=1), np.gradient(seismo_y[:,0,shift:], dt, axis=1), np.gradient(seismo_z[:,0,shift:], dt,     axis=1), moment[0][shift:]
    else:
        return seismo_x[:,0,shift:], seismo_y[:,0,shift:], seismo_z[:,0,shift:], moment[0][shift:]


def force_synthetic(SOURCE_TYPE, force, height, dt, stationPos, sourceParams, mediumParams, 
                        WAVE='BOTH', deriv='DIS', SOURCE_FILTER=False):
    '''
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
    '''
    sourceDim, sourcePos = sourceParams
    mu, rho_rock = mediumParams
    lame = mu # poisson ratio = 1/4
    
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
    nyq_freq = 0.5 / dt # in Hz
    cutoff_freq = 0.03 # in Hz
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
        body_x, body_y, body_z = bw.displacement_force(force, separation, gamma, dt, [rho_rock, lame, mu])
        seismo_x += body_x
        gc.collect()
        seismo_y += body_y
        gc.collect()
        seismo_z += body_z
        gc.collect()
    if WAVE == 'SURF' or WAVE == 'BOTH':
        surf_r, surf_z = sw.rayleigh_displacement_force(-force, stationPos_cyl, np.array([-sourcePos[2]]), dt, [rho_rock, lame, mu])
        seismo_z += surf_z
        gc.collect()
        surf_x, surf_y = cartesian(surf_r, stationPos_cyl)
        gc.collect()
        seismo_x += surf_x
        gc.collect()
        seismo_y += surf_y
        gc.collect()

    if deriv == 'ACC':
        return np.gradient(np.gradient(seismo_x[:,0,shift:], dt, axis=1), dt, axis=1), np.gradient(np.gradient(seismo_y[:,0,shift:], dt, axis=1),     dt, axis=1), np.gradient(np.gradient(seismo_z[:,0,shift:], dt, axis=1), dt, axis=1), force[0][shift:]
    elif deriv == 'VEL':
        return np.gradient(seismo_x[:,0,shift:], dt, axis=1), np.gradient(seismo_y[:,0,shift:], dt, axis=1), np.gradient(seismo_z[:,0,shift:], dt,     axis=1), force[0][shift:]
    else:
        return seismo_x[:,0,shift:], seismo_y[:,0,shift:], seismo_z[:,0,shift:], force[0][shift:]

