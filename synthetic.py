#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import gc

import helpers as hp
import source_setup as ss
import bodywaves_functions as bw
import surfacewaves_functions as sw


# In[ ]:


def translate(new_origin, pos, chamberCent, pistonCent):
    '''
    transforms the original position vectors with new origin
    
    ---INPUTS---
    new_origin  : (3)             : position of new origin in old coordinates
    pos         : (# stations, 3) : station positions in old coordinates
    chamberCent : (3)             : chamber centroid in old coordinates
    pistonCent  : (3)             : piston centroid in old coordinates
    ---RETURNS---
    new_pos     : (# stations, 3) : stations positions w.r.t. new origin
    new_chamber : (3)             : chamber centroid w.r.t. new origin
    new_piston  : (3)             : piston centroid w.r.t. new origin
    '''
    nn = pos.shape[0]
    
    new_pos = np.zeros((nn, 3))
    for ii in range(nn):
        new_pos[ii] = pos[ii] - new_origin
        
    new_chamber = chamberCent - new_origin
    new_piston = pistonCent - new_origin
    
    return new_pos, new_chamber, new_piston


# In[ ]:


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
        else:
            phi = np.arctan(original[ii][1] / original[ii][0])
        new[ii] = np.array([np.linalg.norm(original[ii][:2]), phi, -original[ii][2]])
    return new


# In[ ]:


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


# In[ ]:


def synthetic(pressure, shear_force, dt, stationPos, chamberParams, pistonParams, mediumParams, deriv='ACC'):
    '''
    calculates the full point source synthetic seismograms for a piston-chamber system at given station positions
    
    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards
        
        () indicate numpy arrays
        [] indicate lists
    
    ---INPUTS---
    pressure      : (# time points) : chamber pressure history
    shear_force   : (# time points) : shear force history (between piston and wall)
    dt            : (dt)            : time step size (assumes equal time-stepping)
    stationPos    : (# stations, 3) : positions of accelerometer/seismometer stations
    chamberParams : [1, (3)]        : [chamber volume (m^3), centroid position vector]
    pistonParams  : [1, 1, (3)]     : [(piston height (m), piston radius (m)), 
                                       piston source position vector]
    mediumParams  : [2]             : [shear modulus (Pa), rock density (kg/m^3)] (assumes Poisson ratio = 1/4)
    
    deriv         : string          : seismogram time derivative to return
                                      (options: 'ACC' acceleration; 'VEL' velocity; 'DIS' displacement)
    ---RETURNS---
    seismo_x, seismo_y, seismo_z : (# stations, # time points) : chosen deriv applied to synthetic seismograms
    '''
    
    chamber_vol, chamber_centOLD = chamberParams
    piston_height, piston_radius, piston_posOLD = pistonParams
    mu, rho_rock = mediumParams
    lame = mu # poisson ratio = 1/4
    
    shear_area = 2 * np.pi * piston_radius * piston_height
    cross_section = np.pi * piston_radius**2
    
    # transforming to origin over chamber centroid (at surface)
    chamber_origin = np.array([chamber_centOLD[0], chamber_centOLD[1], 0])
    pos, chamber_cent, piston_cent = translate(chamber_origin, stationPos, chamber_centOLD, piston_posOLD)
    
    # converting position vectors into cylindrical coordinates (to use in SW calc)
    # +z downwards (just for SW calculations)
    pos_cyl = cylindrical(pos)
    
    # storing coordinates for separation vectors between point-source and seismometers
    # separation: (# receivers, # sources)
    # gamma: (# receivers, # sources, 3)
    separationCH, gammaCH = hp.separation_distances_vectors(pos, [chamber_cent])
    separationPI, gammaPI = hp.separation_distances_vectors(pos, [piston_cent])
    gc.collect()
    
    # setting up moment source
    shift = 4000
    moment = ss.moment_density(np.array([pressure]), 0.75 * chamber_vol, cushion=shift)[0]
    moment_tensor = ((lame + 2*mu) / mu) * np.eye(3)
    gc.collect()
    
    # calculating moment contributions
    seismo_x, seismo_y, seismo_z = bw.displacement_moment([moment], moment_tensor, separationCH, 
                                                          gammaCH, dt, [rho_rock, lame, mu])
    r_mom, z_mom = sw.rayleigh_displacement_moment([moment], moment_tensor, pos_cyl, 
                                                   np.array([-chamber_cent[2]]), dt, [rho_rock, lame, mu])
    seismo_z += z_mom
    gc.collect()
    x_mom, y_mom = cartesian(r_mom, pos_cyl)
    gc.collect()
    seismo_x += x_mom
    gc.collect()
    seismo_y += y_mom
    gc.collect()
    
    
    # calculating force contributions
    force_history = (shear_area * shear_force) - (cross_section * pressure)
    force = ss.moment_density(np.array([force_history]), 1, cushion=shift)[0]
    gc.collect()
    
    x_bwf, y_bwf, z_bwf = bw.displacement_force([force], separationPI, gammaPI, dt, [rho_rock, lame, mu])
    seismo_x += x_bwf
    gc.collect()
    seismo_y += y_bwf
    gc.collect()
    seismo_z += z_bwf
    gc.collect()
    
    r_swf, z_swf = sw.rayleigh_displacement_force([-force], pos_cyl, np.array([-piston_cent[2]]), 
                                                  dt, [rho_rock, lame, mu])
    seismo_z += z_swf
    gc.collect()
    x_swf, y_swf = cartesian(r_swf, pos_cyl)
    seismo_x += x_swf
    gc.collect()
    seismo_y += y_swf
    gc.collect()
    
    
    if deriv == 'ACC':
        return np.gradient(np.gradient(seismo_x[:,0,shift:], dt, axis=1), dt, axis=1), np.gradient(np.gradient(seismo_y[:,0,shift:], dt, axis=1), dt, axis=1), np.gradient(np.gradient(seismo_z[:,0,shift:], dt, axis=1), dt, axis=1)
    elif deriv == 'VEL':
        return np.gradient(seismo_x[:,0,shift:], dt, axis=1), np.gradient(seismo_y[:,0,shift:], dt, axis=1), np.gradient(seismo_z[:,0,shift:], dt, axis=1)
    else:
        return seismo_x[:,0,shift:], seismo_y[:,0,shift:], seismo_z[:,0,shift:]


# In[ ]:




