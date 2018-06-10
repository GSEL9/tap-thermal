# -*- coding: utf-8 -*-

# This module is part of tapThermal.

# The module contains the analytical model of the transient diffusion process
# across a one-zone reactor exclusive the temperature gradient.

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np


def one_zone_analytics(dimensional=True, adsorption=False, n_iter=10, **kwargs):
    """Analytical models of dimensional and dimensionless TAP standard
    diffusion, and diffusion with reversible adsorption.

    Args:
        time (array of floats): Time points (s)
        D (float): Diffusion coefficient (m^2/s)
        L (float): Reactor length (m)
        eps (float): Fractional voidage of the packed bed in the reactor
        Np (float): Number of moles or molecules in the inlet pulse (mol)
        ka (float): Adsorption rate constant(m^3/mol-s)
        kd (float): Desorption rate constant (1/s)
        a_s (float): Surface concentration of active sites (mol/m^2)
        Sv (float): Surf area of catalyst / volume catalyst (1/m)

    Kwargs:
        diffusion_only (bool): Call to standard diffusion model
        dimensionless (bool): Call to dimensionless model
        x (float of array of floats): Spatial coordinate(s) (m)

    Returns:
        ndarray: Gas flux (mol/s) as reactor exit.

    """

    time = np.array(kwargs['time'], dtype=float)
    space = np.array(kwargs['space'], dtype=float)

    # Axial dimension parameters.
    try:
        length = kwargs['length']

    except:
        length = float(space[-1])

    try:
        axial = kwargs['axial_loc']

        if isinstance(axial, (int, float)):
            axial_loc = float(axial)
        else:
            axial_loc = np.array(axial, dtype=float)

    except:
        axial_loc = float(length)

    #Initialize solution array
    if isinstance(axial_loc, np.ndarray):
        solu = np.zeros([time.size, axial_loc.size])

    else:
        solu = np.zeros(time.size)

    num_molecules = float(kwargs['num_molecules'])
    eps, coeff = float(kwargs['eps']), float(kwargs['coeff'])

    if dimensional:
        # Dimensional models

        if adsorption:
            ads_rate = float(kwargs['ads_rate'])
            des_rate = float(kwargs['des_rate'])
            surf_area = float(kwargs['surf_area'])
            active_sites = float(kwargs['active_sites'])

            dim_ads_rate = surf_area * active_sites * (1.0 - eps) * ads_rate
            dim_des_rate = eps * des_rate

            # Reversible adsorption model
            model = _dimensional_reversible_adsorption(
                solu, time, space, length, axial_loc, coeff, eps, ads_rate,
                des_rate, surf_area, active_sites, n_iter=n_iter
            )

        else:
            # Standard diffusion model
            model = _dimensional_standard_diffusion(
                solu, time, space, length, axial_loc, coeff, eps, n_iter
            )

    else:
        # Dimensionless models

        tau = time * coeff / (eps * length ** 2)

        if adsorption:
            ads_rate = float(kwargs['ads_rate'])
            des_rate = float(kwargs['des_rate'])
            surf_area = float(kwargs['surf_area'])
            active_sites = float(kwargs['active_sites'])

            dimless_ads_rate = ads_rate * eps * length ** 2 / coeff
            dimless_des_rate = des_rate * eps * length ** 2 / coeff

            model = _dimensionless_reversible_adsorption(
                solu, tau, space, length, axial_loc, coeff, eps,
                dimless_ads_rate, dimless_des_rate, surf_area, active_sites,
                n_iter=n_iter
            )

        else:
            # Standard diffusion model
            model = _dimensionless_standard_diffusion(
                solu, tau, space, length, axial_loc, coeff, eps, n_iter
            )

    return model


def _dimensional_standard_diffusion(solu, time, space, *args, n_iter=10):
    # The dimensional standard diffusion analytical model.

    length, axial_loc, coeff, eps = args[0], args[1], args[2], args[3]

    zeta = axial_loc / length
    sigma = np.pi * coeff / (eps * length ** 2)

    for index in range(2, time.size):
        tau = time[index] * sigma * np.pi

        for num in range(n_iter):
            theta = (num + 0.5) * np.pi * zeta
            eta = (-1) * ((num + 0.5) ** 2 * tau)

            solu[index] += (2.0 * num + 1.0) * np.sin(theta) * np.exp(eta)

    return np.array(solu * sigma, dtype=float)


def _dimensionless_standard_diffusion(solu, tau, space, *args, n_iter=10):
    # The dimensional standard diffusion analytical model.

    length, axial_loc, coeff, eps = args[0], args[1], args[2], args[3]

    zeta = axial_loc / length
    for index in range(2, tau.size):

        for num in range(n_iter):
            theta = (num + 0.5) * np.pi * zeta
            eta = (-1) * ((num + 0.5) ** 2 * tau[index]) * np.pi ** 2

            solu[index] += (2.0 * num + 1.0) * np.sin(theta) * np.exp(eta)

    return np.array(solu * np.pi)


def _dimensional_reversible_adsorption(solu, time, space, *args, n_iter=10):
    # The dimensional standard diffusion with reversible adsorption analytical
    # model.

    length, axial_loc, coeff, eps = args[0], args[1], args[2], args[3]
    ads_rate, des_rate, surf_area = args[4], args[5], args[6]
    active_sites = args[7]

    # Parameter equations
    zeta = axial_loc / length
    chi = length ** 2 / coeff
    sigma = 1.0 / chi / eps

    for index in range(2, time.size):

        for num in range(n_iter):

            # Parameter equations
            param_pn = ((num + 0.5) * np.pi) ** 2

            rho = param_pn + ads_rate * chi + des_rate * chi
            phi = rho ** 2 - 4 * param_pn * des_rate * chi

            param_rp = (-0.5) * (rho) + np.sqrt(phi)
            param_rm = (-0.5) * (rho) - np.sqrt(phi)
            param_diff = (param_rp - param_rm)

            param_an = (param_rp + param_pn + ads_rate * chi) / param_diff

            theta = (num + 0.5) * np.pi * zeta
            psi = param_an * np.exp(param_rm * time[index] * sigma)
            kappa = (1.0 - param_an) * np.exp(param_rp * time[index] * sigma)

            solu[index] += (2 * num + 1) * np.sin(theta) * psi + kappa

    return np.array(solu * np.pi * sigma, dtype=float)


def _dimensionless_reversible_adsorption(solu, time, space, *args, n_iter=10):
    # The dimensional standard diffusion with reversible adsorption analytical
    # model.

    length, axial_loc, coeff, eps = args[0], args[1], args[2], args[3]
    ads_rate, des_rate, surf_area = args[4], args[5], args[6]
    active_sites = args[7]

    # Parameter equations
    chi = length ** 2 / coeff
    sigma = coeff / (eps * length ** 2)

    for index in range(2, time.size):

        for num in range(n_iter):

            # Parameter equations
            param_pn = ((num + 0.5) * np.pi) ** 2
            rho = param_pn + ads_rate + des_rate
            phi = rho ** 2 - 4 * param_pn * des_rate

            param_rp = (-0.5) * (rho) + np.sqrt(phi)
            param_rm = (-0.5) * (rho) - np.sqrt(phi)

            param_an = (param_rp + param_pn + ads_rate) / (param_rp - param_rm)

            theta = (num + 0.5) * np.pi * chi
            psi = param_an * np.exp(param_rm * time[index] * sigma)
            kappa = (1.0 - param_an) * np.exp(param_rp * time[index] * sigma)

            solu[index] += (2 * num + 1) * np.sin(theta) * psi + kappa

    return np.array(solu * np.pi * sigma, dtype=float)
