#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gc


# In[3]:


def p_wave_ff_hat(M_hat, M_ij, omega, r, c, rho, gamma):
    '''
    returns the spectrum of far-field p-wave given FT of source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    
    returns retarded wave in frequency space
    '''
    prefactor = 0
    if c != 0 and r != 0:
        prefactor = np.exp(-1j * omega * r / c) / (4 * np.pi * rho * r * c**3)
    p_hat = 1j * omega * prefactor * M_hat
    
    # summing over directional cosines to get polarization for the p-wave
    ggd = 0
    for ii in range(3):
        for jj in range(3):
            ggd += gamma[ii] * gamma[jj] * M_ij[ii, jj]
            
    # displacement in the x direction from contributions from impulses in each direction
    p_hat_x = p_hat * ggd * gamma[0]
    p_hat_y = p_hat * ggd * gamma[1]
    p_hat_z = p_hat * ggd * gamma[2]
    
    
    return p_hat_x, p_hat_y, p_hat_z


# In[4]:


def s_wave_ff_hat(M_hat, M_ij, omega, r, c, rho, gamma):
    '''
    returns the spectrum of the FAR-field s-wave given FT of source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    '''
    prefactor = 0
    if c != 0 and r != 0:
        prefactor = np.exp(-1j * omega * r / c) / (4 * np.pi * rho * r * c**3)
    s_hat = 1j * omega * prefactor * M_hat
    
    # summing over directional cosines to get polarization for the s-wave
    pol = np.zeros(3)
    for ii in range(3):
        for jj in range(3):
            pol[ii] += gamma[jj] * M_ij[ii, jj]
            for kk in range(3):
                pol[ii] += - gamma[ii] * gamma[jj] * gamma[kk] * M_ij[jj, kk]
    
    # displacement in the x direction from contributions from impulses in each direction
    s_hat_x = s_hat * pol[0]
    s_hat_y = s_hat * pol[1]
    s_hat_z = s_hat * pol[2]
    
    return s_hat_x, s_hat_y, s_hat_z


# In[5]:


def p_wave_if_hat(M_hat, M_ij, omega, r, c, rho, gamma):
    '''
    returns the spectrum of the INTERMEDIATE-field P-wave given FT of source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    '''
    prefactor = 0
    if c != 0 and r != 0:
        prefactor = np.exp(-1j * omega * r / c) / (4 * np.pi * rho * r**2 * c**2)
    p_hat = prefactor * M_hat
    
    # summing over directional cosines to get polarization for the s-wave
    pol = np.zeros(3)
    for ii in range(3):
        for jj in range(3):
            pol[ii] += - gamma[ii] * M_ij[jj, jj]
            pol[ii] += - gamma[jj] * M_ij[jj, ii]
            pol[ii] += - gamma[jj] * M_ij[ii, jj]
            for kk in range(3):
                pol[ii] += 6 * gamma[ii] * gamma[jj] * gamma[kk] * M_ij[jj, kk]
    
    # displacement in the x direction from contributions from impulses in each direction
    p_hat_x = p_hat * pol[0]
    p_hat_y = p_hat * pol[1]
    p_hat_z = p_hat * pol[2]
    
    return p_hat_x, p_hat_y, p_hat_z


# In[6]:


def s_wave_if_hat(M_hat, M_ij, omega, r, c, rho, gamma):
    '''
    returns the spectrum of the INTERMEDIATE-field S-wave given FT of source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    '''
    prefactor = 0
    if c != 0 and r != 0:
        prefactor = np.exp(-1j * omega * r / c) / (4 * np.pi * rho * r**2 * c**2)
    s_hat = prefactor * M_hat
    
    # summing over directional cosines to get polarization for the s-wave
    pol = np.zeros(3)
    for ii in range(3):
        for jj in range(3):
            pol[ii] += gamma[ii] * M_ij[jj, jj]
            pol[ii] += gamma[jj] * M_ij[jj, ii]
            pol[ii] += 2 * gamma[jj] * M_ij[ii, jj]
            for kk in range(3):
                pol[ii] += - 6 * gamma[ii] * gamma[jj] * gamma[kk] * M_ij[jj, kk]
    
    # displacement in the x direction from contributions from impulses in each direction
    s_hat_x = s_hat * pol[0]
    s_hat_y = s_hat * pol[1]
    s_hat_z = s_hat * pol[2]
    
    return s_hat_x, s_hat_y, s_hat_z


# In[7]:


def nearfield_hat(M_hat, M_ij, omega, r, c_p, c_s, rho, gamma):
    '''
    returns the spectrum of the NEAR-field radiation given FT of source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    '''
    # adding nonzero to omega = 0 so there's no runtime error
    # prefactor at omega = 0 is treated separately
    zero = np.where(omega == 0)[0]
    cheat = np.zeros(len(omega))
    cheat[zero] = 1
    
    prefactor = 0
    if r != 0:
        if c_s != 0:
            prefactor = (1 + 1j * omega * r / c_s) * np.exp(-1j * omega * r / c_s)
        if c_p != 0:
            prefactor += - (1 + 1j * omega * r / c_p) * np.exp(-1j * omega * r / c_p)
        prefactor *= 1 / (4 * np.pi * rho * r**4 * (omega + cheat)**2)
        prefactor[zero] = ((r / c_s)**2 - (r / c_p)**2) / (8 * np.pi * rho * r**4)
    nf_hat = prefactor * M_hat
    
    # summing over directional cosines to get polarization for the radiation
    pol = np.zeros(3)
    for ii in range(3):
        for jj in range(3):
            pol[ii] += - 3 * gamma[ii] * M_ij[jj, jj]
            pol[ii] += - 3 * gamma[jj] * M_ij[jj, ii]
            pol[ii] += - 3 * gamma[jj] * M_ij[ii, jj]
            for kk in range(3):
                pol[ii] += 15 * gamma[ii] * gamma[jj] * gamma[kk] * M_ij[jj, kk]
    
    # displacement in the x direction from contributions from impulses in each direction
    nf_hat_x = nf_hat * pol[0]
    nf_hat_y = nf_hat * pol[1]
    nf_hat_z = nf_hat * pol[2]
    
    return nf_hat_x, nf_hat_y, nf_hat_z


# In[8]:


def displacement_moment(moment, moment_tensor, separation, gamma, dt, mediumParams,                        terms = 'all', ps_tuner = 'pON-sON'):
    '''
    displacement with option to turn on/off p and s waves and different radiation terms
    defaults to: all radiation terms calculated, both p and s waves
    
    NOTE: near field term will NOT be split into p and s contributions (best to have 'pON-sON'
            when calculating near field terms)
    
    moment          : (# sources, # time points)  : moment time series at each source point
    moment_tensor   : (3,3)                       : unit moment tensor (defined in cartesian coordinates)
    separation      : (# receivers, # sources)    : magnitude of separation
    gamma           : (# receivers, # sources, 3) : unit separation vectors (source to receiver)
    dt              : (1)                         : time step size
    mediumParams    : (3)                         : (rho, lame, mu)
    
    terms string    : (options: 'all', 'near', 'intermediate', 'far', 'near+intermediate')
    ps_tuner string : (options: 'pON-sON', 'pON-sOFF', 'pOFF-sON')
    '''
    rho, lame, mu = mediumParams
    c_p = np.sqrt((lame + 2 * mu) / rho)
    c_s = np.sqrt(mu / rho)
    
    PON = False
    SON = False
    if (ps_tuner == 'pON-sON') or (ps_tuner == 'pON-sOFF'):
        PON = True
    if (ps_tuner == 'pON-sON') or (ps_tuner == 'pOFF-sON'):
        SON = True
        
    NEAR = False
    INTERMEDIATE = False
    FAR = False
    if (terms == 'near') or (terms == 'near+intermediate') or (terms == 'all'):
        NEAR = True
    if (terms == 'intermediate') or (terms == 'near+intermediate') or (terms == 'all'):
        INTERMEDIATE = True
    if (terms == 'far') or (terms == 'all'):
        FAR = True
    
    nn = np.ma.size(separation, axis=0)
    HH = np.ma.size(separation, axis=1)
    NN = np.ma.size(moment, axis=1)
    omega = np.fft.fftfreq(2*NN, dt) * (2 * np.pi)
    
    
    # to deal with edge effects of an aperiodic function, creating periodic function to use for DFT
    periodic_moment = np.concatenate((moment, np.flip(moment, axis=1)), axis=1)
    
    gc.collect()
    
    moment_hat = np.fft.fft(periodic_moment, axis=1) * dt
    
    gc.collect()
    
    # p-wave spectrum for each seismometer
    p_hat_x = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    p_hat_y = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    p_hat_z = np.zeros((nn, HH, 2*NN), dtype = 'complex')

    if PON:
        if FAR:
            for ii in range(nn):
                for jj in range(HH):
                    p_hat_x[ii,jj,:], p_hat_y[ii,jj,:], p_hat_z[ii,jj,:] = p_wave_ff_hat(moment_hat[jj], moment_tensor, omega, separation[ii,jj], c_p, rho, gamma[ii,jj])
        if INTERMEDIATE:
            for ii in range(nn):
                for jj in range(HH):
                    x, y, z = p_wave_if_hat(moment_hat[jj], moment_tensor, omega, separation[ii,jj], c_p, rho, gamma[ii,jj])    
                    p_hat_x[ii,jj,:] += x
                    gc.collect()
                    p_hat_y[ii,jj,:] += y
                    gc.collect()
                    p_hat_z[ii,jj,:] += z
                    gc.collect()
    
    displacement_hat_x = p_hat_x
    gc.collect()
    displacement_hat_y = p_hat_y
    gc.collect()
    displacement_hat_z = p_hat_z
    gc.collect()

    
    # s-wave spectrum for each seismometer
    s_hat_x = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    s_hat_y = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    s_hat_z = np.zeros((nn, HH, 2*NN), dtype = 'complex')

    if SON:
        if FAR:
            for ii in range(nn):
                for jj in range(HH):
                    s_hat_x[ii,jj,:], s_hat_y[ii,jj,:], s_hat_z[ii,jj,:] = s_wave_ff_hat(moment_hat[jj], moment_tensor, omega, separation[ii,jj], c_s, rho, gamma[ii,jj])
        if INTERMEDIATE:
            for ii in range(nn):
                for jj in range(HH):
                    x1, y1, z1 = s_wave_if_hat(moment_hat[jj], moment_tensor, omega, separation[ii,jj], c_s, rho, gamma[ii,jj])
                    s_hat_x[ii,jj,:] += x1
                    gc.collect()
                    s_hat_y[ii,jj,:] += y1
                    gc.collect()
                    s_hat_z[ii,jj,:] += z1
                    gc.collect()
    
    displacement_hat_x += s_hat_x
    gc.collect()
    displacement_hat_y += s_hat_y
    gc.collect()
    displacement_hat_z += s_hat_z
    gc.collect()

        
    if NEAR:
        nf_hat_x = np.zeros((nn, HH, 2*NN), dtype = 'complex')
        nf_hat_y = np.zeros((nn, HH, 2*NN), dtype = 'complex')
        nf_hat_z = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    
        for ii in range(nn):
            for jj in range(HH):
                nf_hat_x[ii,jj,:], nf_hat_y[ii,jj,:], nf_hat_z[ii,jj,:] = nearfield_hat(moment_hat[jj], moment_tensor, omega, separation[ii,jj], c_p, c_s, rho, gamma[ii,jj])
        gc.collect()
        displacement_hat_x += nf_hat_x
        gc.collect()
        displacement_hat_y += nf_hat_y
        gc.collect()
        displacement_hat_z += nf_hat_z
        gc.collect()

    # inverse fourier transform full displacement back to time domain
    # only care about first half of result
    displacement_x = np.fft.ifft(displacement_hat_x,axis=-1)[:,:,:NN] / dt
    gc.collect()
    displacement_y = np.fft.ifft(displacement_hat_y,axis=-1)[:,:,:NN] / dt
    gc.collect()
    displacement_z = np.fft.ifft(displacement_hat_z,axis=-1)[:,:,:NN] / dt
    gc.collect()

    return displacement_x, displacement_y, displacement_z


# In[1]:


def p_singleforce_ff_hat(F_hat, omega, r, c, rho, gamma):
    '''
    returns spectrum of FAR-field P-wave given FT of SINGLE-FORCE source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    
    RIGHT NOW: only vertical force considered 
    '''
    prefactor = 0
    if c != 0 and r != 0:
        prefactor = np.exp(-1j * omega * r / c) / (4 * np.pi * rho * r * c**2)
    p_hat = prefactor * F_hat
    
    # summing over directional cosines to get polarization for the s-wave
    pol = np.zeros(3)
    for ii in range(3):
        pol[ii] += gamma[ii] * gamma[2]
    
    # displacement in the x direction from contributions from impulses in each direction
    p_hat_x = p_hat * pol[0]
    p_hat_y = p_hat * pol[1]
    p_hat_z = p_hat * pol[2]
    
    return p_hat_x, p_hat_y, p_hat_z


# In[2]:


def s_singleforce_ff_hat(F_hat, omega, r, c, rho, gamma):
    '''
    returns spectrum of FAR-field s-wave given FT of SINGLE-FORCE source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    
    RIGHT NOW: only vertical force considered
    '''
    prefactor = 0
    if c != 0 and r != 0:
        prefactor = np.exp(-1j * omega * r / c) / (4 * np.pi * rho * r * c**2)
    s_hat = prefactor * F_hat
    
    # summing over directional cosines to get polarization for the s-wave
    pol = np.zeros(3)
    for ii in range(3):
        pol[ii] += - gamma[ii] * gamma[2]
        if ii == 2:
            pol[ii] += 1
    
    # displacement in the x direction from contributions from impulses in each direction
    s_hat_x = s_hat * pol[0]
    s_hat_y = s_hat * pol[1]
    s_hat_z = s_hat * pol[2]
    
    return s_hat_x, s_hat_y, s_hat_z


# In[3]:


def singleforce_nf_hat(F_hat, omega, r, c_p, c_s, rho, gamma):
    '''
    returns spectrum of NEAR-field s-wave given FT of SINGLE-FORCE source
    and the unit vector position of the receiver, gamma
    splits into x, y, and z displacements
    
    RIGHT NOW: only vertical force considered
    '''
    # adding nonzero to omega = 0 so there's no runtime error
    # prefactor at omega = 0 is treated separately
    zero = np.where(omega == 0)[0]
    cheat = np.zeros(len(omega))
    cheat[zero] = 1
    
    prefactor = 0
    if r != 0:
        if c_s != 0:
            prefactor = (1 + 1j * omega * r / c_s) * np.exp(-1j * omega * r / c_s)
        if c_p != 0:
            prefactor += - (1 + 1j * omega * r / c_p) * np.exp(-1j * omega * r / c_p)
        prefactor *= 1 / (4 * np.pi * rho * r**3 * (omega + cheat)**2)
        prefactor[zero] = ((r / c_s)**2 - (r / c_p)**2) / (8 * np.pi * rho * r**3)
    nf_hat = prefactor * F_hat
    
    # summing over directional cosines to get polarization for the radiation
    pol = np.zeros(3)
    for ii in range(3):
        pol[ii] += 3 * gamma[ii] * gamma[2]
        if ii == 2:
            pol[ii] += -1
    
    # displacement in the x direction from contributions from impulses in each direction
    nf_hat_x = nf_hat * pol[0]
    nf_hat_y = nf_hat * pol[1]
    nf_hat_z = nf_hat * pol[2]
    
    return nf_hat_x, nf_hat_y, nf_hat_z


# In[4]:


def displacement_force(force, separation, gamma, dt, mediumParams,                        terms = 'all', ps_tuner = 'pON-sON'):
    '''
    displacement with option to turn on/off p and s waves and different radiation terms
    defaults to: all radiation terms calculated, both p and s waves
    
    NOTE: near field term will NOT be split into p and s contributions (best to have 'pON-sON'
            when calculating near field terms)
            
    RIGHT NOW: only vertical force considered (positive == upwards)
    
    force           : (# sources, # time points)  : vertical single force time series at each source point
    separation      : (# receivers, # sources)    : magnitude of separation
    gamma           : (# receivers, # sources, 3) : unit separation vectors (source to receiver)
    dt              : (1)                         : time step size
    mediumParams    : (3)                         : (rho, lame, mu)
    
    terms string    : (options: 'all', 'near', 'far')
    ps_tuner string : (options: 'pON-sON', 'pON-sOFF', 'pOFF-sON')
    '''
    rho, lame, mu = mediumParams
    c_p = np.sqrt((lame + 2 * mu) / rho)
    c_s = np.sqrt(mu / rho)
    
    PON = False
    SON = False
    if (ps_tuner == 'pON-sON') or (ps_tuner == 'pON-sOFF'):
        PON = True
    if (ps_tuner == 'pON-sON') or (ps_tuner == 'pOFF-sON'):
        SON = True
        
    NEAR = False
    FAR = False
    if (terms == 'near') or (terms == 'all'):
        NEAR = True
    if (terms == 'far') or (terms == 'all'):
        FAR = True
    
    nn = np.ma.size(separation, axis=0)
    HH = np.ma.size(separation, axis=1)
    NN = np.ma.size(force, axis=1)
    omega = np.fft.fftfreq(2*NN, dt) * (2 * np.pi)
    
    # to deal with edge effects of an aperiodic function, creating periodic function to use for DFT
    periodic_force = np.concatenate((force, np.flip(force, axis=1)), axis=1)
    
    force_hat = np.fft.fft(periodic_force, axis=1) * dt
    
    gc.collect()
    
    # p-wave spectrum for each seismometer
    p_hat_x = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    p_hat_y = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    p_hat_z = np.zeros((nn, HH, 2*NN), dtype = 'complex')

    if PON:
        if FAR:
            for ii in range(nn):
                for jj in range(HH):
                    p_hat_x[ii,jj,:], p_hat_y[ii,jj,:], p_hat_z[ii,jj,:] = p_singleforce_ff_hat(force_hat[jj], omega, separation[ii,jj], c_p, rho, gamma[ii,jj])
    
    # s-wave spectrum for each seismometer
    s_hat_x = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    s_hat_y = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    s_hat_z = np.zeros((nn, HH, 2*NN), dtype = 'complex')

    if SON:
        if FAR:
            for ii in range(nn):
                for jj in range(HH):
                    s_hat_x[ii,jj,:], s_hat_y[ii,jj,:], s_hat_z[ii,jj,:] = s_singleforce_ff_hat(force_hat[jj], omega, separation[ii,jj], c_s, rho, gamma[ii,jj])
    
    gc.collect()
    
    # combining wavetypes
    displacement_hat_x = p_hat_x + s_hat_x
    displacement_hat_y = p_hat_y + s_hat_y
    displacement_hat_z = p_hat_z + s_hat_z
    
    gc.collect()
    
    if NEAR:
        nf_hat_x = np.zeros((nn, HH, 2*NN), dtype = 'complex')
        nf_hat_y = np.zeros((nn, HH, 2*NN), dtype = 'complex')
        nf_hat_z = np.zeros((nn, HH, 2*NN), dtype = 'complex')
    
        for ii in range(nn):
            for jj in range(HH):
                nf_hat_x[ii,jj,:], nf_hat_y[ii,jj,:], nf_hat_z[ii,jj,:] = singleforce_nf_hat(force_hat[jj], omega, separation[ii,jj], c_p, c_s, rho, gamma[ii,jj])
        
        displacement_hat_x += nf_hat_x
        displacement_hat_y += nf_hat_y
        displacement_hat_z += nf_hat_z
        
        gc.collect()

    # inverse fourier transform full displacement back to time domain
    # only care about first half of result
    displacement_x = np.fft.ifft(displacement_hat_x,axis=-1)[:,:,:NN] / dt
    displacement_y = np.fft.ifft(displacement_hat_y,axis=-1)[:,:,:NN] / dt
    displacement_z = np.fft.ifft(displacement_hat_z,axis=-1)[:,:,:NN] / dt
    
    gc.collect()

    return displacement_x, displacement_y, displacement_z


# In[ ]:




