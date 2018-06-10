# -*- coding: utf-8 -*-

# This module is part of tapThermal.

# The module contains a function producing the representation of a temperature
# gradient along the microreactor.

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np


def _check_iterable(value, label):
    """Raises TypeError if value is not iterable."""

    if not isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError('{} must be <numpy.ndarray>, <list> or <tuple>, and not'
                        '{}'.format(label, type(value)))


def thermal_gradient(spatial_grid, temperatures):
    """Distributes give temperature measurements along the reactor as local
    piecewise constant functions representing the temperature gradient."""

    _check_iterable(spatial_grid, label='spatial grid')
    _check_iterable(temperatures, label='temperatures')

    grid = np.array(spatial_grid, dtype=float)
    measurements = np.array(temperatures, dtype=float)

    # Divides original grid into sections.
    section_grid = np.linspace(grid[0], grid[-1], measurements.size)

    conditions = []
    for num, _ in enumerate(section_grid[:-1]):
        conditions.append(
            (grid >= section_grid[num]) & (grid <= section_grid[num + 1])
        )

    return np.piecewise(grid, conditions, temperatures)
