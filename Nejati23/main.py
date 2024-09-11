import cvxpy as cp
import numpy as np
from fossil import domains
from experiments.scenapp_tests.benchmarks import models

def Jet_engine(N):
    c = 0
    L_f = 4.71
    M_f = 3.16
    max_rhoP = 0.6
    L_g = 8.06
    lambd = 3.1
    beta = 0.01
    tau = 1e-5
    delta = 1.2438e-4

    T = 1 # ignore this
    mu = -0 # Not sure what to set this to
    
    XD = domains.Rectangle([0.1, 0.1], [1, 1])
    XI = domains.Rectangle([0.1, 0.1], [0.5, 0.5])
    XU = domains.Rectangle([0.7, 0.7], [1, 1])
    init_dom_data = XI._generate_data(500)()
    unsafe_dom_data = XU._generate_data(500)()
    
    init_data = XI._generate_data(N)()

    system = models.JetEngBarr
    all_data = system().generate_trajs(init_data, tau)
    state_data = np.vstack([elem[0] for elem in all_data[1]])
    next_states = np.vstack([elem[1] for elem in all_data[1]])
    
    A_mat = cp.Variable((2,2), symmetric=True)
    B_mat = cp.Variable((2,1))
    C_mat = cp.Variable(1)

    constraints = [A_mat[0,0] == 0, A_mat[1,1] == 0, -0.2 <= A_mat[0,1], A_mat[0,1] <= 0.2, 
                    -0.4 <= B_mat, B_mat <= 0.4, 
                    -0.4 <= C_mat, C_mat <=4]

    eta = cp.Variable(1)
    objective = cp.Minimize(eta)

    gamma = cp.Variable(1)
    constraints.append(gamma + c*T - lambd - mu <= eta)

    constraints.append(cp.diag(init_dom_data@A_mat@init_dom_data.T)+(init_dom_data@B_mat).flatten()+C_mat-gamma <= eta)
    constraints.append(-(cp.diag(unsafe_dom_data@A_mat@unsafe_dom_data.T)+(unsafe_dom_data@B_mat).flatten()+C_mat)+lambd <= eta)
    
    constraints.append(-(cp.diag(state_data@A_mat@state_data.T)+(state_data@B_mat).flatten() 
                        - cp.diag(next_states@A_mat@next_states.T)+(next_states@B_mat).flatten())/tau-c+delta <= eta)

    prob = cp.Problem(objective, constraints)
    prob.solve()
    print(eta.value)
    print(B_mat.value[0])
    print(A_mat.value[0,1]*2)
    print(B_mat.value[1])
    print(C_mat.value)
    print(gamma.value)

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
        
