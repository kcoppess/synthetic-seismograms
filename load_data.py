import numpy as np
import gc
from zipfile import ZipFile
import scipy.interpolate as si
import scipy.integrate as sint
import io
import matplotlib.pyplot as plt


def moment_ZIP_load(ZIPFILE, SOURCE_TYPE, TOTAL_TIME, dt):
    """
    loads pressure time series and applies signal processing (smoothing and constant
    time-stepping)

    ---INPUTS---
    ZIPFILE    : string : full path to data zip file
    SOURCE_TYPE: string : 'CONDUIT' -> load conduit pressure
                          'CHAMBER' -> load chamber pressure
    TOTAL_TIME : 1      : total simulation time
    dt         : 1      : desired time-step size
    ---RETURNS---
    p     : (# sources, # time points) : CONDUIT
            (# time points)            : CHAMBER
    time  : (# time points)
    height: (# sources)                : NB positive z is up and z=0 is conduit bottom
    """
    directory = ZipFile(ZIPFILE, mode='r')

    time2 = np.loadtxt(io.BytesIO(directory.read('time.txt')), delimiter=',')
    t_index = np.argwhere(time2 > TOTAL_TIME-0.001)[0,0]
    time1 = time2[:t_index]
    # in conduit flow code, this takes up to be positive z and
    # bottom of conduit is at z = 0
    height = np.loadtxt(io.BytesIO(directory.read('height.txt')), delimiter=',')

    if SOURCE_TYPE == 'CONDUIT':
        p1 = np.loadtxt(io.BytesIO(directory.read('pressure.txt')), delimiter=',')[:,:t_index]
        p1 = np.real(p1)
    elif SOURCE_TYPE == 'CHAMBER':
        p1 = np.loadtxt(io.BytesIO(directory.read('chamber_pressure.txt')), delimiter=',')[:t_index]
        p1 = np.real(p1)

    directory.close()
    gc.collect()

    modulo = 2
    '''signal processing to smooth out numerical effects (e.g. from downsampling)'''
    length = int(TOTAL_TIME / dt)
    time = np.arange(length) * dt

    time_smooth = si.interp1d(np.arange(len(time1))[::modulo], time1[::modulo], kind='cubic')
    times = time_smooth(np.arange(len(time1))[:-modulo])

    if SOURCE_TYPE == 'CONDUIT':
        smooth = si.interp1d(times, p1[:, :-modulo], kind='cubic', axis=1)
        p = smooth(time)
    elif SOURCE_TYPE == 'CHAMBER':
        smooth = si.interp1d(times, p1[:-modulo], kind='cubic')
        p = smooth(time)

    gc.collect()

    return p, time, height


def force_ZIP_load(ZIPFILE, SOURCE_TYPE, TOTAL_TIME, dt):
    """
    loads force time series and applies signal processing (smoothing and constant
    time-stepping)

    ---INPUTS---
    ZIPFILE    : string : full path to data zip file
    SOURCE_TYPE: string : 'CONDUIT' -> load conduit shear pressure
                          'CHAMBER' -> load chamber pressure
    TOTAL_TIME : 1      : total simulation time
    dt         : 1      : desired time-step size
    ---RETURNS---
    f     : (# sources, # time points) : CONDUIT
            (# time points)            : CHAMBER
    time  : (# time points)
    height: (# sources)                : NB positive z is up and z=0 is conduit bottom
    """
    directory = ZipFile(ZIPFILE, mode='r')

    time2 = np.loadtxt(io.BytesIO(directory.read('time.txt')), delimiter=',')
    t_index = np.argwhere(time2 > TOTAL_TIME-0.000001)[0,0]
    time1 = time2[:t_index]
    # in conduit flow code, this takes up to be positive z and
    # bottom of conduit is at z = 0
    height = np.loadtxt(io.BytesIO(directory.read('height.txt')), delimiter=',')

    if SOURCE_TYPE == 'CONDUIT':
        f1 = np.loadtxt(io.BytesIO(directory.read('wall_trac.txt')), delimiter=',')[:,:t_index]
        f1 = -np.real(f1)
    elif SOURCE_TYPE == 'CHAMBER':
        f1a = np.loadtxt(io.BytesIO(directory.read('chamber_pressure.txt')), delimiter=',')[:t_index]
        rho = np.loadtxt(io.BytesIO(directory.read('density.txt')), delimiter=',')[0,:t_index]
        # remember if conduit velocity is positive -> chamber is 
        # losing mass -> need to use -vel for weight change
        df1b_dt = -np.loadtxt(io.BytesIO(directory.read('velocity.txt')), delimiter=',')[0,:t_index] * rho * 9.8
        f1a = -np.real(f1a)
        df1b_dt = -np.real(df1b_dt)

    directory.close()
    gc.collect()
    
    modulo = 2
    '''signal processing to smooth out numerical effects (e.g. from downsampling)'''
    length = int(TOTAL_TIME / dt)
    time = np.arange(length) * dt

    time_smooth = si.interp1d(np.arange(len(time1))[::modulo], time1[::modulo], kind='cubic')
    times = time_smooth(np.arange(len(time1))[:-modulo])

    if SOURCE_TYPE == 'CONDUIT':
        smooth = si.interp1d(times, f1[:, :-modulo], kind='cubic', axis=1)
        f = smooth(time)
    elif SOURCE_TYPE == 'CHAMBER':
        smooth_a = si.interp1d(times, f1a[:-modulo], kind='cubic')
        fa = smooth_a(time)
        smooth_b = si.interp1d(times, df1b_dt[:-modulo], kind='cubic')
        dfb_dt = smooth_b(time)

        fb = sint.cumtrapz(dfb_dt, x=time, initial=0)
        f = fa + fb

    gc.collect()

    return f, time, height
