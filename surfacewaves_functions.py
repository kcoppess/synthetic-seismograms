#!/usr/bin/env python
# coding: utf-8

# Calculation of surface waves (only fundamental mode)
# 
# CURRENT: Rayleigh wave calculation based off of Aki&Richards Ch 7, homogeneous half-space

import numpy as np
import gc
import scipy.optimize as so
import scipy.interpolate as si


def alpha(phaseVel, waveSpeed):
    return np.sqrt(1 - (phaseVel ** 2 / waveSpeed ** 2), dtype='complex')


def rayleigh_function(phaseVel, c_s, c_p):
    """
    returns values of Rayleigh function, derived from solutions to equations of motion
    and enforcing boundary/radiation conditions for surface waves

    NB: roots give the allowed phase velocities of surface waves in medium
    """
    alphaS = alpha(phaseVel, c_s)
    alphaP = alpha(phaseVel, c_p)

    RR = (4 * alphaS * alphaP - (1 + alphaS ** 2) ** 2)

    return RR


def rayleigh_function_prime(phaseVel, c_s, c_p):
    """
    returns values of gradient of Rayleigh function
    """
    alphaS = alpha(phaseVel, c_s)
    alphaP = alpha(phaseVel, c_p)

    dRR_dc = -4 * (alphaP / alphaS) * (phaseVel / c_s ** 2)
    dRR_dc += -4 * (alphaS / alphaP) * (phaseVel / c_p ** 2)
    dRR_dc += 4 * (1 + alphaS ** 2) * (phaseVel / c_s ** 2)

    return dRR_dc


def find_phaseVel(c_s, c_p):
    """
    returns phase velocity of fundamental mode for surface waves by finding smallest,
    nonzero root of Rayleigh function

    NB: designed for materials with poisson ratio = 1/4 (may lead to issues for other)
    """
    phaseVel = so.root_scalar(rayleigh_function, args=(c_s, c_p), method='newton', fprime=rayleigh_function_prime,
                              x0=0.919 * c_s, xtol=1e-25)
    return phaseVel.root


def rayleigh_eigenfunction_x(z, omega, c_s, c_p):
    """
    returns values of eigenfunction at each z for x-component displacement from
    Rayleigh surface waves of frequency omega

    NB: corresponds to r1 in Aki&Richards Ch7 and +z downwards
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c

    alphaS = alpha(c, c_s)
    alphaP = alpha(c, c_p)
    AA = c_p / c

    r1 = AA * np.exp(-abs(k) * alphaP * z) - 2 * AA * (alphaS * alphaP / (1 + alphaS ** 2)) * np.exp(
        -abs(k) * alphaS * z)
    r1 *= k / abs(k)

    return r1


def rayleigh_eigenfunction_z(z, omega, c_s, c_p):
    """
    returns values of eigenfunction at each z for z-component displacement from
    Rayleigh surface waves of frequency omega

    NB: corresponds to r2 in Aki&Richards Ch7 and +z downwards
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c

    alphaS = alpha(c, c_s)
    alphaP = alpha(c, c_p)
    AA = c_p / c

    r2 = AA * alphaP * np.exp(-abs(k) * alphaP * z) - 2 * AA * (alphaP / (1 + alphaS ** 2)) * np.exp(
        -abs(k) * alphaS * z)

    return r2


def rayleigh_eigenfunction_z_PRIME(z, omega, c_s, c_p):
    """
    returns values of derivative of eigenfunction at each z for z-component displacement from
    Rayleigh surface waves of frequency omega

    NB: corresponds to r2 in Aki&Richards Ch7 and +z downwards
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c

    alphaS = alpha(c, c_s)
    alphaP = alpha(c, c_p)
    AA = c_p / c

    dr2_dz = -abs(k) * AA * alphaP ** 2 * np.exp(-abs(k) * alphaP * z) + 2 * abs(k) * AA * (
                alphaP * alphaS / (1 + alphaS ** 2)) * np.exp(-abs(k) * alphaS * z)

    return dr2_dz


def rayleigh_energy_integral1(rho, omega, c_s, c_p):
    """
    returns value of energy integral I1 for particular Rayleigh wave mode

    NB: from Aki&Richards Ch7, eq 7.74 & 7.144
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c
    alphaS = alpha(c, c_s)
    alphaP = alpha(c, c_p)

    BB = (rho * c_p ** 2 / (2 * c ** 2))

    I1 = BB * (1 + alphaP ** 2) / (2 * abs(k) * alphaP)
    I1 += -BB * 4 * alphaP / (abs(k) * (1 + alphaS ** 2))
    I1 += BB * 2 * alphaP ** 2 / (abs(k) * alphaS * (1 + alphaS ** 2))

    return I1


def rayleigh_energy_integral2(lame, mu, omega, c_s, c_p):
    """
    returns value of energy integral I2 for particular Rayleigh wave mode

    NB: from Aki&Richards Ch7, eq 7.74
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c
    alphaS = alpha(c, c_s)
    alphaP = alpha(c, c_p)

    BB = c_p ** 2 / (2 * c ** 2)
    l2m = lame + 2 * mu

    I2 = BB * (l2m + mu * alphaP ** 2) / (2 * abs(k) * alphaP)
    I2 += - BB * 4 * (alphaP * alphaS * l2m + mu * alphaP ** 2) / (abs(k) * (1 + alphaS ** 2) * (alphaS + alphaP))
    I2 += BB * 2 * alphaP ** 2 * (alphaS ** 2 * l2m + mu) / (abs(k) * alphaS * (1 + alphaS ** 2) ** 2)

    return I2


def rayleigh_energy_integral3(lame, mu, omega, c_s, c_p):
    """
    returns value of energy integral I3 for particular Rayleigh wave mode

    NB: from Aki&Richards Ch7, eq 7.74
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c
    alphaS = alpha(c, c_s)
    alphaP = alpha(c, c_p)

    BB = c_p ** 2 / c ** 2

    I3 = BB * alphaP * (mu - lame) / 2
    I3 += BB * 2 * alphaP * (lame * alphaS * (1 + alphaP ** 2) - mu * alphaP * (1 + alphaS ** 2)) / (
                (alphaP + alphaS) * (1 + alphaS ** 2))
    I3 += -BB * 2 * alphaP ** 2 * alphaS * (lame - mu) / (1 + alphaS ** 2) ** 2
    I3 *= k / abs(k)

    return I3


def rayleigh_group_velocity(rho, lame, mu, omega, c_s, c_p):
    """
    returns Rayleigh group velocity for mode with frequency omega
    """
    c = find_phaseVel(c_s, c_p)
    k = omega / c

    I1 = rayleigh_energy_integral1(rho, omega, c_s, c_p)
    I2 = rayleigh_energy_integral2(lame, mu, omega, c_s, c_p)
    I3 = rayleigh_energy_integral3(lame, mu, omega, c_s, c_p)

    U = (I2 + (I3 / (2 * k))) / (c * I1)

    return U


def rayleigh_displacement_hat_pointforce(omega_np, pos, force_hat, h, mediumParams):
    """
    returns displacement field for Rayleigh waves from VERTICAL POINT FORCE acting at position z=h
    for omega-mode of the force

    NB: ignores omega=0; returns in cylindrical coordinates

    omega_np     : (# frq)    : frequencies of the various modes (NB that sign convention opposite of np.fft)
    pos          : (3)        : position vector for receiver (in cylindrical coordinates: r, phi, z)
    force_hat    : (# frq)    : fourier transformed point force (z-component only)
    h            : (1)        : location of point force
    mediumParams : (3)        : (rho, lame, mu)

    displ_r_hat  : (# frq)
    displ_z_hat  : (# frq)
    """
    omega = -omega_np  # flipping sign to make consistent with np.fft sign convention
    omega[0] = 1  # cheating to deal with division by zero

    rho, lame, mu = mediumParams
    NN = len(omega)

    c_p = np.sqrt((lame + 2 * mu) / rho)
    c_s = np.sqrt(mu / rho)
    c = find_phaseVel(c_s, c_p)
    k = omega / c

    I1 = rayleigh_energy_integral1(rho, omega, c_s, c_p)
    U = rayleigh_group_velocity(rho, lame, mu, omega, c_s, c_p)

    r1_h = rayleigh_eigenfunction_x(h, omega, c_s, c_p)
    r2_h = rayleigh_eigenfunction_z(h, omega, c_s, c_p)
    gc.collect()

    prefactor = force_hat * r2_h
    prefactor *= np.sqrt(2 / (np.pi * k * pos[0])) * np.exp(1j * k * pos[0]) / (8 * c * U * I1)

    displ_r_hat = prefactor * np.ones(NN)
    displ_z_hat = prefactor * np.ones(NN)
    gc.collect()

    displ_r_hat *= np.exp(-1j * np.pi / 4) * rayleigh_eigenfunction_x(pos[2], omega, c_s, c_p)
    displ_z_hat *= np.exp(1j * np.pi / 4) * rayleigh_eigenfunction_z(pos[2], omega, c_s, c_p)

    displ_r_hat[0] = displ_z_hat[0] = 0

    return displ_r_hat, displ_z_hat


def rayleigh_displacement_force(force_z, pos, source_pos, dt, mediumParams):
    """
    returns rayleigh displacement field (in time domain) for each receiver and source
    combination for a (distribution of) point FORCE(s)

    (returns in cylindrical coords)
    NB: calculation done with positive z oriented "downwards" but flip back to conventional
        orientation when returning array
        ALSO force_z is positive z oriented "downwards"

    force_z      : (# sources, # time pts) : vertical single force time series at each source point
    pos          : (# receivers, 3)        : position vectors for receivers (in cylindrical coordinates: r, phi, z)
    source_pos   : (# sources)             : location of point forces along z
    dt           : (1)                     : time step size
    mediumParams : (3)                     : (rho, lame, mu)

    displ_r      : (#receivers, # sources, # time pts)
    displ_z      : (#receivers, # sources, # time pts)
    """
    HH = np.ma.size(force_z, axis=0)  # sources
    NN = np.ma.size(force_z, axis=1)  # time pts
    nn = np.ma.size(pos, axis=0)  # receivers

    #     padding = 10000
    #     ttt = np.arange(2*NN + padding)
    #     ttt_in = np.concatenate((ttt[:NN], ttt[NN+padding:]))
    #     omega = np.fft.fftfreq(2*NN + padding, dt) * (2 * np.pi)

    omega = np.fft.fftfreq(2 * NN, dt) * (2 * np.pi)

    displ_r = np.zeros((nn, HH, NN))
    displ_z = np.zeros((nn, HH, NN))

    for ii in range(HH):
        #         reflect = np.concatenate((force_z[ii], np.flip(force_z[ii])))
        #         # trying to deal with artifact coming from discontinuity in the derivative by smoothing out reflection
        #         smooth = si.interp1d(ttt_in, reflect, kind='cubic')
        #         periodic_force = smooth(ttt)

        periodic_force = np.concatenate((force_z[ii], np.flip(force_z[ii])))
        force_hat = np.fft.fft(periodic_force) * dt
        for jj in range(nn):
            displ_r_hat, displ_z_hat = rayleigh_displacement_hat_pointforce(omega, pos[jj], force_hat, source_pos[ii],
                                                                            mediumParams)
            displ_r[jj, ii] = np.fft.ifft(displ_r_hat)[:NN] / dt
            displ_z[jj, ii] = np.fft.ifft(displ_z_hat)[:NN] / dt

    return displ_r, -displ_z


def rayleigh_displacement_hat_moment(omega_np, pos, moment_hat, moment_tensor, h, mediumParams):
    """
    returns displacement field for Rayleigh waves from POINT DIAGONAL MOMENT SOURCE acting at position z=h
    for omega-mode of the force

    NB: ignores omega=0; returns in cylindrical coordinates

    omega_np     : (# frq)    : frequencies of the various modes
    pos          : (3)        : position vector for receiver (in cylindrical coordinates: r, phi, z)
    moment_hat   : (# frq)    : fourier transformed moment
    moment_tensor: (3, 3)     : unit moment tensor (defined in cartesian coords; DIAGONAL ONLY)
    h            : (1)        : location of point force
    mediumParams : (3)        : (rho, lame, mu)

    displ_r_hat  : (# frq)
    displ_z_hat  : (# frq)
    """
    omega = -omega_np  # flipping sign to make consistent with np.fft sign convention
    omega[0] = 1  # cheating to deal with division by zero

    rho, lame, mu = mediumParams
    NN = len(omega)

    c_p = np.sqrt((lame + 2 * mu) / rho)
    c_s = np.sqrt(mu / rho)
    c = find_phaseVel(c_s, c_p)
    k = omega / c
    k[0] = 1  # cheating to deal with division by zero

    I1 = rayleigh_energy_integral1(rho, omega, c_s, c_p)
    U = rayleigh_group_velocity(rho, lame, mu, omega, c_s, c_p)

    r1_h = rayleigh_eigenfunction_x(h, omega, c_s, c_p)
    r2_h = rayleigh_eigenfunction_z(h, omega, c_s, c_p)
    dr2_dz_h = rayleigh_eigenfunction_z_PRIME(h, omega, c_s, c_p)
    gc.collect()

    prefactor = k * r1_h * (moment_tensor[0, 0] * np.cos(pos[1]) ** 2 + moment_tensor[1, 1] * np.sin(pos[1]) ** 2)
    prefactor += dr2_dz_h * moment_tensor[2, 2]
    prefactor *= moment_hat * np.sqrt(2 / (np.pi * k * pos[0])) * np.exp(1j * k * pos[0]) / (8 * c * U * I1)

    displ_r_hat = prefactor * np.ones(NN)
    displ_z_hat = prefactor * np.ones(NN)
    gc.collect()

    displ_r_hat *= np.exp(-1j * np.pi / 4) * rayleigh_eigenfunction_x(pos[2], omega, c_s, c_p)
    displ_z_hat *= np.exp(1j * np.pi / 4) * rayleigh_eigenfunction_z(pos[2], omega, c_s, c_p)

    displ_r_hat[0] = displ_z_hat[0] = 0

    return displ_r_hat, displ_z_hat


def rayleigh_displacement_moment(moment, moment_tensor, pos, source_pos, dt, mediumParams):
    """
    returns rayleigh displacement field (in time domain) for each receiver and source
    combination for a (distribution of) point MOMENT SOURCE(s)

    (returns in cylindrical coords)
    NB: calculation is done with positive z oriented "downwards" but flip back to conventional
        orientation when returning array

    moment       : (# sources, # time pts) : moment time series at each source point
    moment_tensor: (3, 3)                  : unit moment tensor (defined in cartesian coords; DIAGONAL ONLY)
    pos          : (# receivers, 3)        : position vectors for receivers (in cylindrical coordinates: r, phi, z)
    source_pos   : (# sources)             : location of point forces along z
    dt           : (1)                     : time step size
    mediumParams : (3)                     : (rho, lame, mu)

    displ_r      : (#receivers, # sources, # time pts)
    displ_z      : (#receivers, # sources, # time pts)
    """
    HH = np.ma.size(moment, axis=0)  # sources
    NN = np.ma.size(moment, axis=1)  # time pts
    nn = np.ma.size(pos, axis=0)  # receivers

    padding = 10000
    ttt = np.arange(2 * NN + padding)
    ttt_in = np.concatenate((ttt[:NN], ttt[NN + padding:]))
    omega = np.fft.fftfreq(2 * NN + padding, dt) * (2 * np.pi)

    displ_r = np.zeros((nn, HH, NN))
    displ_z = np.zeros((nn, HH, NN))

    for ii in range(HH):
        gc.collect()
        reflect = np.concatenate((moment[ii], np.flip(moment[ii])))
        # trying to deal with artifact coming from discontinuity in the derivative by smoothing out reflection
        smooth = si.interp1d(ttt_in, reflect, kind='cubic')
        periodic_moment = smooth(ttt)

        # periodic_moment = np.concatenate((moment[ii], np.flip(moment[ii])))
        moment_hat = np.fft.fft(periodic_moment) * dt
        for jj in range(nn):
            displ_r_hat, displ_z_hat = rayleigh_displacement_hat_moment(omega, pos[jj], moment_hat, moment_tensor,
                                                                        source_pos[ii], mediumParams)
            displ_r[jj, ii] = np.fft.ifft(displ_r_hat)[:NN] / dt
            displ_z[jj, ii] = np.fft.ifft(displ_z_hat)[:NN] / dt

    return displ_r, -displ_z

