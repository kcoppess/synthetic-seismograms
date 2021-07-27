#!/usr/bin/env python
# coding: utf-8

# Helper functions for synthetic seismogram calculation

# In[1]:


import numpy as np
import gc


# In[2]:


def separation_distances_vectors(receiver, source):
    """
    takes the positions of the sources and receivers and returns
    array of distances between every combination of source/receiver
    and array of unit separation vectors

    requires:
    source (# sources, 3)
    receiver (# receivers, 3)

    returns:
    separation (# receivers, # sources)
    unit_separation (# receivers, # sources, 3)
    """
    receiver_num = np.ma.size(receiver, axis=0)
    source_num = np.ma.size(source, axis=0)

    separation = np.zeros((receiver_num, source_num))
    unit_separation = np.zeros((receiver_num, source_num, 3))

    for ii in range(receiver_num):
        for jj in range(source_num):
            separation[ii, jj] = np.linalg.norm(receiver[ii] - source[jj])
            if separation[ii, jj] != 0:
                unit_separation[ii, jj] = (receiver[ii] - source[jj]) / separation[ii, jj]

    return separation, unit_separation


# In[3]:


def integration_trapezoid(points, dx, dy=None, dz=None):
    """
    performs numerical integration using trapezoid rule with
    unitform mesh (integrating over axis=1 or segments along tube)

    NOTE: - inputs are required to have 3 axes

    dimensions:
    grid #sources
    dx, dy, dz (# receivers, #sources, # time points)
    """
    diff_dim = dx.shape
    dim = (diff_dim[0], diff_dim[2])
    HH = diff_dim[1]

    grid = np.tile(points, (diff_dim[2], 1)).transpose()
    gc.collect()

    if dy is None:
        final_x = np.zeros(dim, dtype='complex')

        for ii in range(dim[0]):
            final_x[ii, :] = 0.5 * np.sum(np.absolute(grid[1:] - grid[:-1]) * (dx[ii, 1:, :] + dx[ii, :-1, :]), axis=0)
        gc.collect()
        return final_x

    if dz is None:
        final_x = np.zeros(dim, dtype='complex')
        final_y = np.zeros(dim, dtype='complex')

        for ii in range(dim[0]):
            final_x[ii, :] = 0.5 * np.sum(np.absolute(grid[1:] - grid[:-1]) * (dx[ii, 1:, :] + dx[ii, :-1, :]), axis=0)
            gc.collect()
            final_y[ii, :] = 0.5 * np.sum(np.absolute(grid[1:] - grid[:-1]) * (dy[ii, 1:, :] + dy[ii, :-1, :]), axis=0)
            gc.collect()
        return final_x, final_y

    else:
        final_x = np.zeros(dim, dtype='complex')
        final_y = np.zeros(dim, dtype='complex')
        final_z = np.zeros(dim, dtype='complex')

        for ii in range(dim[0]):
            final_x[ii, :] = 0.5 * np.sum(np.absolute(grid[1:] - grid[:-1]) * (dx[ii, 1:, :] + dx[ii, :-1, :]), axis=0)
            gc.collect()
            final_y[ii, :] = 0.5 * np.sum(np.absolute(grid[1:] - grid[:-1]) * (dy[ii, 1:, :] + dy[ii, :-1, :]), axis=0)
            gc.collect()
            final_z[ii, :] = 0.5 * np.sum(np.absolute(grid[1:] - grid[:-1]) * (dz[ii, 1:, :] + dz[ii, :-1, :]), axis=0)
            gc.collect()

        return final_x, final_y, final_z


# In[4]:


def cartesian_to_spherical(x, y, z, pos):
    """
    given the cartesian x, y, and z waveforms, project onto the
    radial, phi, and theta directions for a given location
    labeled by vector pos (from origin)

    see 'Del in cylindrical and spherical coordinates' Wikipedia for relevant
    transformation rules to find spherical unit vectors
    NOTE: for vertical projection, unit vector is not defined along z-axis (no angle in x-y plane)

    r: radial
    phi: transverse (angle in x-y plane: 0 < phi < 2pi); positive direction oriented "into page"
    theta: vertical (angle wrt z axis: -pi/2 < theta < pi/2; positive direction oriented "downwards"
    """
    r = np.linalg.norm(pos)
    r_cyl = np.linalg.norm(pos[:2])

    radial = np.zeros(len(x))
    phi = np.zeros(len(x))
    theta = np.zeros(len(x))

    if r != 0:
        radial = (x * pos[0] + y * pos[1] + z * pos[2]) / r
    if r_cyl != 0:
        phi = (- x * pos[1] + y * pos[0]) / r_cyl
    if r != 0 and r_cyl != 0:
        theta = ((x * pos[0] + y * pos[1]) * pos[2] - (r_cyl) ** 2 * z) / (r * r_cyl)

    return radial, phi, theta


# In[5]:


def cartesian_to_cylindrical(x, y, z, pos):
    """
    given the cartesian x, y, and z waveforms, project onto the
    radial, transverse, and vertical directions for a given location
    labeled by vector pos (from origin)

    see 'Del in cylindrical and spherical coordinates' Wikipedia for relevant
    transformation rules to find cylindrical unit vectors
    NOTE: for vertical projection, unit vector is not defined along z-axis (no angle in x-y plane)

    r: radial
    phi: transverse (angle in x-y plane: 0 < phi < 2pi); positive direction oriented "into page"
    z: vertical ; positive direction oriented "upwards" (same as typical Cartesian z)
    """
    r_cyl = np.linalg.norm(pos[:2])

    radial = np.zeros(len(x))
    transverse = np.zeros(len(x))
    vertical = z

    if r_cyl != 0:
        radial = (x * pos[0] + y * pos[1]) / r_cyl
        transverse = (- x * pos[1] + y * pos[0]) / r_cyl

    return radial, transverse, vertical


# In[6]:


def logistic(x, k, L):
    """
    returns the values of a logistic function (at points x)
    with growth rate k and maximum value L
    """
    return L / (1 + np.exp(-k * x))


# In[7]:


def tube_pressure_smoothed(t, Dpressure, vel, z, shift):
    """
    returns smoothed retarded function for given time for particular height
    along tube
    """
    pres = np.zeros(len(t))
    if vel != 0:
        pres = logistic((t - shift) - (z / vel), 1, Dpressure)

    return pres


# In[8]:


def seismogram_plot_setup(axs, plot_title, spine_width=1, spine_color='grey'):
    """
    sets up the plot formatting for seismograms (sharing x-axis)
    """
    axs[0].set_title(plot_title)

    for ax in axs:
        ax.spines["top"].set_linewidth(spine_width)
        ax.spines["right"].set_linewidth(spine_width)
        ax.spines["bottom"].set_linewidth(spine_width)
        ax.spines["left"].set_linewidth(spine_width)

        ax.spines["top"].set_color(spine_color)
        ax.spines["right"].set_color(spine_color)
        ax.spines["bottom"].set_color(spine_color)
        ax.spines["left"].set_color(spine_color)

        ax.tick_params(axis='both', length=4, width=1)
        ax.axhline(0, color='lightgrey', linestyle='--', linewidth=spine_width)

    return axs

# In[ ]:




