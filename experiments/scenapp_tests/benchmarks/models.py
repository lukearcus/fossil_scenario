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

class Barr4D(control.DynamicalModel):
    n_vars = 4
    time_horizon = 4

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x1, x2, x3, x4 = v[0], v[1], v[2], v[3]
        else:
            x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return [x1 + 0.2 * x1 * x2 -0.5*x3*x4, np.cos(x4), 0.01*np.sqrt(np.abs(x1)), -x1 - x2**2 + np.sin(x4)]

class Barr4D_DT(control.DynamicalModel):
    n_vars = 4
    time_horizon = 40
    time="discrete"
    T=0.1

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x1, x2, x3, x4 = v[0], v[1], v[2], v[3]
        else:
            x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return [x1+self.T*(x1 + 0.2 * x1 * x2 -0.5*x3*x4), x2+self.T*(np.cos(x4)), x3+self.T*0.01*np.sqrt(np.abs(x1)), x4+self.T*(-x1 - x2**2 + np.sin(x4))]

class DC_Motor(control.DynamicalModel):
    n_vars = 2
    time_horizon = 100
    T = 0.01
    time = "discrete"
    
    def f_torch(self, t, v):
        R =1
        L=0.01
        J=0.01
        b=1
        kdc=0.01

        if len(v.shape) == 1:
            x1, x2= v[0], v[1]
        else:
            x1, x2= v[:, 0], v[:, 1]

        return [x1+self.T*(((-R/L)*x1)-((kdc/L)*x2)), x2+self.T*(((kdc/J)*x1)-((b/J)*x2))]


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

class HighOrd8DT(control.DynamicalModel):
    n_vars = 8
    time_horizon = 50
    time = "discrete"
    T=0.1

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
            x0+self.T*x1,
            x1+self.T*x2,
            x2+self.T*x3,
            x3+self.T*x4,
            x4+self.T*x5,
            x5+self.T*x6,
            x6+self.T*x7,
            x7 + self.T*(-20 * x7
            - 170 * x6
            - 800 * x5
            - 2273 * x4
            - 3980 * x3
            - 4180 * x2
            - 2400 * x1
            - 576 * x0),
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

class JetEngBarrDT(control.DynamicalModel):
    n_vars = 2
    time_horizon = 500
    time = "discrete"
    T=0.01

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [x+self.T*(-y-1.5*x**2-0.5*x**3), y+self.T*x]

class Circle(control.DynamicalModel):
    n_vars = 2
    time_horizon = 50
    time = "discrete"

    def f_torch(self, t, v):
        
        if len(v.shape) == 1:
            x1, x2 = v[0], v[1]
        else:
            x1, x2 = v[:, 0], v[:, 1]

        return [x1-x2, 0.5*x2+0.5*x1]

class Spiral(control.DynamicalModel):
    n_vars = 2
    time_horizon = 50
    time = "discrete"
    T=0.5

    def f_torch(self, t, v):
        T=self.T
        if len(v.shape) == 1:
            x1, x2 = v[0], v[1]
        else:
            x1, x2 = v[:, 0], v[:, 1]

        return [x1-T*x2, x2+T*(x1-x2)]

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

class ThreeRoomTemp(control.DynamicalModel):
    # from Data-Driven Verification and synthesis of
    # Stochastic Systems via Barrier Certificates
    # itself adapted from Girard et al, 2016,
    # Safety controller synthesis for incrementally stable switched
    # systems using multiscale symbolic models.
    n_vars = 3
    time_horizon = 3
    time = "discrete" 
    
    tau = 5  # discretise param
    alpha = 6.2 * 1e-3  # heat exchange
    alpha_e = 8 * 1e-3  # heat exchange 1
    temp_e = 10  # external temp

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x1, x2, x3 = v[0], v[1], v[2]
        else:
            x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]

        q1 = (
            (1 - self.tau * (self.alpha + self.alpha_e)) * x1
            + self.tau * self.alpha * x2
            + self.tau * self.alpha_e * self.temp_e + np.random.normal(scale=0.01)
        )
        q2 = (
            (1 - self.tau * (2*self.alpha + self.alpha_e)) * x2
            + self.tau * self.alpha * (x1+x3)
            + self.tau * self.alpha_e * self.temp_e+ np.random.normal(scale=0.01)
        )
        q3 = (
            (1 - self.tau * (self.alpha + self.alpha_e)) * x3
            + self.tau * self.alpha * (x2)
            + self.tau * self.alpha_e * self.temp_e+ np.random.normal(scale=0.01)
        )

        return [q1, q2, q3]

class SecondOrderLQR(control.DynamicalModel):
    n_vars = 2
    K = [1.0, 1.73]
    time_horizon =500 

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y - x**3, u1]


class ThirdOrderLQR(control.DynamicalModel):
    n_vars = 3
    K = [23.71, 18.49, 0.0]
    time_horizon = 2

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x1, x2, x3 = v[0], v[1], v[2]
        else:
            x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1 = -(self.K[0] * x1 + self.K[1] * x2 + self.K[2] * x3)
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]
