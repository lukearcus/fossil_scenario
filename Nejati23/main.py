import cvxpy as cp
import numpy as np

def Jet_engine(N):
    c = 0
    L_f = 4.71
    M_f = 3.16
    max_rhoP = 0.6
    L_g = 8.06
    lambd = 3.1
    beta = 0.01

class RCP_SCP:
    def __init__(self, barrier_func, barrier_lie_func, barr_data_params):
        self.barr = barrier_func
        self.barr_lie = barrier_lie_func
        self.data_params = barr_data_params

    def solve(self, data):
        self.data_params.value = data
        gamma = cp.Variable(1)
        lambd = cp.Variable(1)
        c = cp.Variable(1)
        
