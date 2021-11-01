import numpy as np
import gc
from zipfile import ZipFile
import scipy.interpolate as si
import io


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
    length = int(TOTAL_TIME / dt)
    time = np.arange(length) * dt

    time_smooth = si.interp1d(np.arange(len(time1))[::2], time1[::2], kind='cubic')
    times = time_smooth(np.arange(len(time1))[:-2])

    if SOURCE_TYPE == 'CONDUIT':
        smooth = si.interp1d(times, p1[:, :-2], kind='cubic', axis=1)
        p = smooth(time)
    elif SOURCE_TYPE == 'CHAMBER':
        smooth = si.interp1d(times, p1[:-2], kind='cubic')
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
    length = int(TOTAL_TIME / dt)
    time = np.arange(length) * dt

    time_smooth = si.interp1d(np.arange(len(time1))[::2], time1[::2], kind='cubic')
    times = time_smooth(np.arange(len(time1))[:-2])

    if SOURCE_TYPE == 'CONDUIT':
        smooth = si.interp1d(times, f1[:, :-2], kind='cubic', axis=1)
        f = smooth(time)
    elif SOURCE_TYPE == 'CHAMBER':
        smooth = si.interp1d(times, f1[:-2], kind='cubic')
        f = smooth(time)

    gc.collect()

    return f, time, height
