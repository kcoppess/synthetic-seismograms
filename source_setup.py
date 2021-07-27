#!/usr/bin/env python
# coding: utf-8

# Sets up the source description (both single force and moment) from results of conduit flow model

# In[2]:


import numpy as np
import gc


# In[ ]:


def time_height(time, dt, height, cushion=4000):
    """
    takes the time and height arrays from conduit flow simulation and
    puts in form for calculations to fit

    NB: - cushion is adding time points prior to simulation start (to eliminate some
          edge effects from fourier transform)
        - relabelling positions along conduit so that z=0 is the top
    """
    NN = len(time)
    t = np.arange(-cushion, NN) * dt
    t[cushion:] = time

    z = -(height - max(height) * np.ones(len(height)))

    return t, z


# In[3]:


def equivforce_per_unitvolume(time, density_vel):
    """
    returns the equivalent single force per unit volume acting on surrounding earth
    (positive -> upwards)

    (based off of methods in Sec 4, Takei & Kumazawa 1993)

    density_vel : (# grid points, # time points)
    df_dV       : (# grid points, # time points)
    """
    dim = density_vel.shape
    df_dV = np.zeros(dim, dtype='float')

    # d(rho * v)/dt
    df_dV += - np.gradient(density_vel, time, axis=1)

    return df_dV


# In[ ]:


def singleforce_density(density_vel, time, area, cushion=4000):
    """
    returns dforce/dz time series for each point along conduit with added
    cushion at the beginning, given rho_v time series for each point

    NB: returns the change in dforce/dz from the initial value (looking at deviations
        from original equilibrium state)
        (positive -> upwards : ACTING ON SURROUNDING EARTH)

    density_vel : (# grid points, # OG time points) : rho*v time series at each grid point
    time        : (# OG time points + cushion)      : cushioned time points
    area        : (1)                               : conduit cross-sectional area
    cushion     : (1)                               : # time points of constant cushion
    """
    HH = np.ma.size(density_vel, axis=0)
    NN_og = np.ma.size(density_vel, axis=1)
    NN = NN_og + cushion

    rho_v = np.zeros((HH, NN))
    ones = np.ones(cushion)
    for ii in range(HH):
        if cushion != 0:
            rho_v[ii, :cushion] = ones * density_vel[ii, 0]
        rho_v[ii, cushion:] = density_vel[ii, :]
    gc.collect()

    dforce_dz_true = area * equivforce_per_unitvolume(time, rho_v)
    dforce_dz_initial = np.tile(dforce_dz_true[:, 0], (NN, 1)).transpose()
    dforce_dz = dforce_dz_true - dforce_dz_initial
    gc.collect()

    return dforce_dz


# In[1]:


def moment_tensor_cylindricalSource(mediumParams):
    """
    returns the moment tensor for a cylindrical source oriented along the z-axis
    NB: defined in cartesian basis
    """
    lame, mu = mediumParams

    moment_tensor = ((lame + 2 * mu) / mu) * np.eye(3)
    moment_tensor[2, 2] = lame / mu

    return moment_tensor


# In[2]:


def moment_density(pressure, area, cushion=4000):
    """
    returns dmoment/dz time series for each point along conduit with added
    cushion at the beginning, given pressure time series for each point

    NB: returns the change in dmoment/dz from the initial value (looking at deviations
        from original equilibrium state)

    pressure : (# grid points, # OG time points) : pressure time series at each grid point
    area     : (1)                               : conduit cross-sectional area
    cushion  : (1)                               : # time points of constant cushion
    """
    HH = np.ma.size(pressure, axis=0)
    NN_og = np.ma.size(pressure, axis=1)
    NN = NN_og + cushion

    dp_cushion = np.zeros((HH, NN))
    for ii in range(HH):
        dp_cushion[ii, cushion:] = pressure[ii, :] - pressure[ii, 0] * np.ones(NN_og)
    gc.collect()

    dmoment_dz = area * dp_cushion

    return dmoment_dz

