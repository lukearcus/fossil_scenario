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
    time_horizon = 0.5 
    stochastic = True

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]

    def g_torch(self, t, v):
        return np.diag([0.1,0.1])

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]

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

class JetEngBarr(control.DynamicalModel):
    n_vars = 2
    time_horizon = 5

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [-y-1.5*x**2-0.5*x**3, x]

class TwoRoomTemp(control.DynamicalModel):
    # from Data-Driven Safety Verification of
    # Stochastic Systems via Barrier Certificates
    # itself adapted from Girard et al, 2016,
    # Safety controller synthesis for incrementally stable switched
    # systems using multiscale symbolic models.
    n_vars = 2
    time_horizon = 10
    time = "discrete" 
    
    tau = 5  # discretise param
    alpha = 5 * 1e-2  # heat exchange
    alpha_e1 = 5 * 1e-3  # heat exchange 1
    alpha_e2 = 8 * 1e-3  # heat exchange 2
    temp_e = 15  # external temp
    alpha_h = 3.6 * 1e-3  # heat exchange room-heater
    temp_h = 55  # boiler temp

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x1, x2 = v[0], v[1]
        else:
            x1, x2 = v[:, 0], v[:, 1]

        q1 = (
            (1 - self.tau * (self.alpha + self.alpha_e1)) * x1
            + self.tau * self.alpha * x2
            + self.tau * self.alpha_e1 * self.temp_e
        )
        q2 = (
            (1 - self.tau * (1.0 * self.alpha + self.alpha_e2)) * x2
            + self.tau * self.alpha * (x1)
            + self.tau * self.alpha_e2 * self.temp_e
        )

        return [q1, q2]
