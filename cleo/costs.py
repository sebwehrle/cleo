"""Neutral cost helper ownership for cross-layer reuse."""

from __future__ import annotations

import numpy as np


def turbine_overnight_cost(power, hub_height, rotor_diameter, year):
    """Estimate turbine overnight investment cost in EUR per kW."""
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    spec_power = power * 10 ** 6 / rotor_area
    cost = ((620 * np.log(hub_height)) - (1.68 * spec_power) + (182 * (2016 - year) ** 0.5) - 1005)
    return cost.astype("float")
