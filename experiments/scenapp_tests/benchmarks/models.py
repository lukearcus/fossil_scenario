import sys
import inspect
from typing import Any

import numpy as np
import torch

from matplotlib import pyplot as plt

from fossil import control

class Barr1(control.DynamicalModel):
    n_vars = 2
    time_horizon = 2

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]

class Barr1_stoch(control.DynamicalModel):
    n_vars = 2
    time_horizon = 2

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [y + 2 * x * y + np.random.normal(loc=-0.25), -x - y**2 + 2 * x**2+np.random.normal(loc=-1)]

class HighOrd8(control.DynamicalModel):
    n_vars = 8
    time_horizon = 2

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x0, x1, x2, x3, x4, x5, x6, x7 = (
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
            )
        
        else:
            x0, x1, x2, x3, x4, x5, x6, x7 = (
                v[:, 0],
                v[:, 1],
                v[:, 2],
                v[:, 3],
                v[:, 4],
                v[:, 5],
                v[:, 6],
                v[:, 7],
            )
        return [
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            -20 * x7
            - 170 * x6
            - 800 * x5
            - 2273 * x4
            - 3980 * x3
            - 4180 * x2
            - 2400 * x1
            - 576 * x0,
        ]
