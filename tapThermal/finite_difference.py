# -*- coding: utf-8 -*-

# This module is part of tapThermal.

# The module contains functions representing the finite difference expressions
# of the model equation at each grid point.

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


def inlet(y, index, dydt, sigma, initial_condition):
    """Reactor inlet as intial grid point."""

    backward = 1 / 3 * (4 * y[index] - y[index + 1] + initial_condition)

    dydt[index] = sigma * (y[index + 1] - 2 * y[index] + backward)

    return dydt


def zone_one(y, index, dydt, sigma):
    """Inner grid points of zone one."""

    dydt[index] = sigma * (y[index + 1] - 2 * y[index] + y[index - 1])

    return dydt


def zone_one_final(y, index, dydt, sigma, eta_one, eta_two):
    """Final grid point of zone one."""

    backward = 4 * eta_one * y[index] - eta_one * y[index - 1]
    forward = 4 * eta_two * y[index + 2] - eta_two * y[index + 3]

    zone_final = (backward + forward) / (3 * eta_one + 3 * eta_two)

    dydt[index] = sigma * (zone_final - 2 * y[index] + y[index - 1])

    return dydt


def boundary_two(y, index, dydt, sigma, eta_one, eta_two):
    """Boundary between zone one and zone two."""

    backward = 4 * eta_one * y[index - 1] - eta_one * y[index - 2]
    forward = 4 * eta_two * y[index + 1] - eta_two * y[index + 2]

    boundary = (backward + forward) / (3 * eta_one + 3 * eta_two)

    dydt[index] = sigma * (y[index + 1] - 2 * boundary + y[index - 1])

    return dydt

def zone_two_initial(y, index, dydt, sigma, eta_one, eta_two):
    """Initial grid point of zone two."""

    backward = 4 * eta_one * y[index - 2] - eta_one * y[index - 3]
    forward = 4 * eta_two * y[index + 2] - eta_two * y[index + 3]

    zone_initial = (backward + forward) / (3 * eta_one + 3 * eta_two)

    dydt[index] = sigma * (y[index + 1] - 2 * y[index] + zone_initial)

    return dydt


def zone_two(y, index, dydt, sigma):
    """Inner grid points of zone two."""

    dydt[index] = sigma * (y[index + 1] - 2 * y[index] + y[index - 1])

    return dydt


def zone_two_final(y, index, dydt, sigma, eta_two, eta_three):
    """Final grid point of zone two."""

    backward = 4 * eta_two * y[index] - eta_two * y[index - 1]
    forward = 4 * eta_three * y[index + 2] - eta_three * y[index + 3]

    zone_final = (3 * eta_two + 3 * eta_three)

    dydt[index] = sigma * (zone_final - 2 * y[index] + y[index - 1])

    return dydt


def boundary_three(y, index, dydt, sigma, eta_two, eta_three):
    """Boundary between zone two and zone three."""

    backward = 4 * eta_one * y[index - 1] - eta_one * y[index - 2]
    forward = 4 * eta_two * y[index + 1] - eta_two * y[index + 2]

    boundary = (backward + forward) / (3 * eta_two + 3 * eta_three)

    dydt[index] = sigma * (y[index + 1] - 2 * boundary + y[index - 1])

    return dydt


def zone_three_initial(y, index, dydt, sigma, eta_two, eta_three):
    """Initial grid point of zone three."""

    backward = 4 * eta_two * y[index - 2] - eta_two * y[index - 3]
    forward = 4 * eta_three * y[index + 2] - eta_three * y[index + 3]

    zone_initial = (backward + forward) / (3 * eta_two + 3 * eta_three)

    dydt[index] = sigma * (y[index + 1] - 2 * y[index] + zone_initial)

    return dydt


def zone_three(y, index, dydt, sigma):
    """Inner grid points of zone three."""

    dydt[index] = sigma * (y[index + 1] - 2 * y[index] + y[index - 1])

    return dydt


def outlet(y, index, dydt, sigma):
    """Final grid point of zone three."""

    dydt[index] = sigma * (-2 * y[index] + y[index - 1])

    return dydt
