from __future__ import annotations

import shapely.geometry as sg


def square(x0=0.0, y0=0.0, size=1.0):
    return sg.box(x0, y0, x0 + size, y0 + size)
