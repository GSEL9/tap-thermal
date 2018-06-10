# -*- coding: utf-8 -*-

# This module is part of tapThermal.

# The module contains a function producing the representation of a temperature
# gradient along the microreactor.

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np

from scipy.integrate import odeint


def knudsen_diffusion_coeff(temp, ref_coeff, mass=40.0, **kwargs):
    """Knudsen diffusion coefficient"""

    try:
        ref_temp = kwargs['ref_temp']
    except:
        ref_temp = 298.15

    try:
        ref_mass = kwargs['ref_mass']
    except:
        ref_mass = 40.0

    return ref_coeff * np.sqrt(temp * ref_mass / (ref_temp * mass))


def boundary_cond(time, num_molecules=1.e-9, tau=1.e-3):
    """Boundary condition of diffusion model"""

    # NOTE: Aviod overflow
    exp = np.clip(np.exp(time / tau), 1.e-12, 1.e12)

    return (num_molecules * time / (tau ** 2)) / exp


def _one_zone_fd(y, time, *args, eps=0.4):
    """Converts the one-zone diffusion PDE model to ODE by finite difference
    scheme."""

    ref_coeff = float(args[0])

    gradient = np.array(args[1], dtype=float)

    gridpoints, step_size, area = int(args[2]), float(args[3]), float(args[4])

    # Solution vector
    dydt = np.zeros(gridpoints)
    # Parameter coefficient
    phi = step_size ** 2  * eps

    # Reactor inlet
    for loc in [1]:
        coeff = knudsen_diffusion_coeff(gradient[loc], ref_coeff)

        initial_condition = 2 * step_size * boundary_cond(time) / (coeff * area)
        backward = 1 / 3 * (4 * y[loc] - y[loc + 1] + initial_condition)

        dydt[loc] = coeff / phi * (y[loc + 1] - 2 * y[loc] + backward)

    # Internal grid
    for loc in range(2, gridpoints - 2):
        sigma = knudsen_diffusion_coeff(gradient[loc], ref_coeff) / phi

        dydt[loc] = sigma * (y[loc + 1] - 2 * y[loc] + y[loc - 1])

    # Reactor outlet
    for loc in [gridpoints - 2]:
        sigma = knudsen_diffusion_coeff(gradient[loc], ref_coeff) / phi

        dydt[loc] = sigma * ((-2) * y[loc] + y[loc - 1])

    return dydt


def _three_zone_fd(y, time, *args, eps=0.4):
    """Converts the three-zone diffusion PDE model to ODE by finite differences.
    """

    ref_coeffs = np.array(args[0], dtype=float)
    gradient = np.array(args[1], dtype=float)
    grid = np.array(args[2], dtype=int)

    step_size, area = float(args[3]), float(args[4])

    # Solution vector
    dydt = np.zeros(int(np.sum(grid)))
    # Parameter coefficient
    phi = step_size ** 2  * eps

    # Reactor inlet
    for loc in [int(1)]:
        coeff = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[0])

        initial_condition = 2 * step_size * boundary_cond(time) / (coeff * area)
        backward = 1 / 3 * (4 * y[loc] - y[loc + 1] + initial_condition)

        dydt[loc] = coeff / phi * (y[loc + 1] - 2 * y[loc] + backward)

    # Internal grid zone one
    for loc in range(int(2), int(grid[0] - 1)):
        sigma = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[0]) / phi

        dydt[loc] = sigma * (y[loc + 1] - 2 * y[loc] + y[loc - 1])

    # Final grid point in zone one
    for loc in [int(grid[0] - 1)]:
        coeff_z_one = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[0])
        coeff_z_two = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])

        backward = 4 * coeff_z_one * y[loc] - coeff_z_one * y[loc - 1]
        forward = 4 * coeff_z_two * y[loc + 2] - coeff_z_two * y[loc + 3]
        fin_zone = (backward + forward) / (3 * coeff_z_one + 3 * coeff_z_two)

        dydt[loc] = coeff_z_one / phi * (fin_zone - 2 * y[loc] + y[loc - 1])

    # Boundary between zone one and two
    for loc in [int(grid[0])]:
        coeff_z_one = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[0])
        coeff_z_two = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])

        backward = 4 * coeff_z_one * y[loc - 1] - coeff_z_one * y[loc - 2]
        forward = 4 * coeff_z_two * y[loc + 1] - coeff_z_two * y[loc + 2]
        boundary = (backward + forward) / (3 * coeff_z_one + 3 * coeff_z_two)

        dydt[loc] = coeff_z_one / phi * (y[loc + 1] - 2 * boundary + y[loc - 1])

    # Initial grid point in zone two
    for loc in [grid[0] + 1]:
        coeff_z_one = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[0])
        coeff_z_two = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])

        backward = 4 * coeff_z_one * y[loc - 2] - coeff_z_one * y[loc - 3]
        forward = 4 * coeff_z_two * y[loc] - coeff_z_two * y[loc + 1]
        init_zone = (backward + forward) / (3 * coeff_z_one + 3 * coeff_z_two)

        dydt[loc] = coeff_z_two / phi * (y[loc + 1] - 2 * y[loc] + init_zone)

    # Internal grid zone two
    for loc in range (int(grid[0] + 2), int(np.sum(grid[:2]) - 1)):
        coeff = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])

        dydt[loc] = coeff / phi * (y[loc + 1] - 2 * y[loc] + y[loc - 1])

    # Final grid point in zone two
    for loc in [int(np.sum(grid[:2]) - 1)]:
        coeff_z_two = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])
        coeff_z_three = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[2])

        backward = 4 * coeff_z_two * y[loc] - coeff_z_two * y[loc - 1]
        forward = 4 * coeff_z_three * y[loc + 2] - coeff_z_three * y[loc + 3]
        fin_zone = (backward + forward) / (3 * coeff_z_two + 3 * coeff_z_three)

        dydt[loc] = coeff_z_two / phi * (fin_zone - 2 * y[loc] + y[loc - 1])

    # Boundary between zone two and three
    for loc in [int(np.sum(grid[:2]))]:
        coeff_z_two = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])
        coeff_z_three = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[2])

        backward = 4 * coeff_z_two * y[loc - 1] - coeff_z_two * y[loc - 2]
        forward = 4 * coeff_z_three * y[loc + 1] - coeff_z_three * y[loc + 2]
        boundary = (backward + forward) / (3 * coeff_z_two + 3 * coeff_z_three)

        dydt[loc] = coeff_z_two / phi * (y[loc + 1] - 2 * boundary + y[loc - 1])

    # Initial grid point of zone three
    for loc in [int(np.sum(grid[:2]) + 1)]:
        coeff_z_two = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[1])
        coeff_z_three = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[2])

        backward = 4 * coeff_z_two * y[loc - 2] - coeff_z_two * y[loc - 3]
        forward = 4 * coeff_z_three * y[loc] - coeff_z_three * y[loc + 1]
        init_zone = (backward + forward) / (3 * coeff_z_two + 3 * coeff_z_three)

        dydt[loc] = coeff_z_three / phi * (y[loc + 1] - 2 * y[loc] + init_zone)

    # Internal grid of zone three
    for loc in range(int(np.sum(grid[:2]) + 2), int(np.sum(grid) - 2)):
        coeff = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[2])

        dydt[loc] = coeff / phi  * (y[loc + 1] - 2 * y[loc] + y[loc - 1])

    for loc in [int(np.sum(grid) - 2)]:
        coeff = knudsen_diffusion_coeff(gradient[loc], ref_coeffs[2])

        dydt[loc] = coeff / phi * (-2 * y[loc] + y[loc - 1])

    return dydt


def one_zone_numerical(params, ref_coeff, num_molecules=1e-9):
    """Returns one zone reactor exit flow."""

    time = np.array(params[0], dtype=float)
    gradient = np.array(params[1], dtype=float)

    gridpoints = int(params[2])

    step_size, area = float(params[3]), float(params[4])

    solu = odeint(
        _one_zone_fd, np.zeros(int(gradient.size)), time,
        args=(ref_coeff, gradient, gridpoints, step_size, area)
    )

    return solu[:, -2] * ref_coeff * area / (step_size * num_molecules)


def three_zone_numerical(params, ref_coeffs, num_molecules=1.e-9):
    """Returns three zone reactor exit flow."""

    time = np.array(params[0], dtype=float)
    gradient = np.array(params[1], dtype=float)

    grid = np.array(params[2], dtype=int)

    step_size, area = float(params[3]), float(params[4])

    solu = odeint(
        _three_zone_fd, np.zeros(int(gradient.size)), time,
        args=(ref_coeffs, gradient, grid, step_size, area)
    )

    return solu[:, -2] * ref_coeffs[1] * area / (step_size * num_molecules)
