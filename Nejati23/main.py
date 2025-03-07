import cvxpy as cp
import numpy as np
from fossil import domains
from fossil import plotting
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import *
from scipy.special import betaincinv, betainc
from scipy import stats
from tqdm import tqdm

import torch
torch.manual_seed(0)

class certificate:
    beta = None
    _type = "Nejati23 Barrier"
    name = "Nejati23_Barrier"

    def __init__(self, A, B, c):
        self.A = np.array(A)
        self.B = np.array(B)
        self.c = np.array(c)

    def __call__(self,data):
        data = np.array(data)
        res = np.diag(data@self.A@data.T)+(data@self.B).flatten()+self.c
        return res
    
    def compute_net_gradnet(self, data):
        data = np.array(data)
        net = np.diag(data@self.A@data.T)+(data@self.B).flatten()+self.c
        grad_net = 2*self.A@data.T+self.B
        return net, grad_net

def Jet_engine(N, discrete=False):
    c = 0
    L_f = 4.71
    M_f = 3.16
    max_rhoP = 0.6
    L_g = 8.06
    lambd = 3.1
    beta = 0.01
    tau = 1e-5
    delta = 1.2438e-4
    k = 4+1+1 #q1-4, gamma, and eta
    k = 5 #don't count eta

    eps = betaincinv(k, N-k+1, 1-beta)

    T = 1 # ignore this
    mu = -0.01 # Not sure what to set this to
    
    XD = domains.Rectangle([0.1, 0.1], [1, 1])
    XI = domains.Rectangle([0.1, 0.1], [0.5, 0.5])
    XU = domains.Rectangle([0.7, 0.7], [1, 1])
    
    print("Generating data")
    N_state = 1000
    init_dom_data = XI._generate_data(N_state)()
    unsafe_dom_data = XU._generate_data(N_state)()
    
    init_data = XD._generate_data(N)()

    if discrete:
        system = models.JetEngBarrDT
    else:
        system = models.JetEngBarr
    if discrete:
        all_data = system().generate_trajs(init_data, 1)
    else:
        all_data = system().generate_trajs(init_data, tau)
    state_data = np.vstack([elem[:, 0] for elem in all_data[1]])
    next_states = np.vstack([elem[:, -1] for elem in all_data[1]])
    
    print("Data generation complete")
    print("Building constraints")

    A_mat = cp.Variable((2,2), symmetric=True)
    B_mat = cp.Variable((2,1))
    C_mat = cp.Variable(1)

    constraints = [A_mat[0,0] == 0, A_mat[1,1] == 0, -0.2 <= A_mat[0,1], A_mat[0,1] <= 0.2, 
                    -0.4 <= B_mat, B_mat <= 0.4, 
                    -0.4 <= C_mat, C_mat <=4] # they say this is in [-0.4,0.4] but then find it as 2.7288...

    eta = cp.Variable(1)
    objective = cp.Minimize(eta)

    gamma = cp.Variable(1)
    constraints.append(gamma + c*T - lambd - mu <= eta)
    
    #for elem in init_dom_data:
    #    constraints.append(elem@A_mat@elem+elem@B_mat+C_mat-gamma<=eta)
    #for elem in unsafe_dom_data:
    #    constraints.append(-(elem@A_mat@elem+elem@ B_mat+C_mat)+lambd <= eta)
    #for elem, next_s in zip(state_data, next_states):
    #    constraints.append((next_s@A_mat@next_s+next_s@B_mat-(elem@A_mat@elem+elem@B_mat))/tau-c+delta <= eta)

    for init, unsafe, state, next_s in zip(init_dom_data, unsafe_dom_data, state_data, next_states):
        constraints.append(init@A_mat@init+init@B_mat+C_mat-gamma<=eta)
        constraints.append(-(unsafe@A_mat@unsafe+unsafe@ B_mat+C_mat)+lambd <= eta)
        if discrete:
            constraints.append((next_s@A_mat@next_s+next_s@B_mat-(state@A_mat@state+state@B_mat))/tau-c <= eta)
        else:
            constraints.append((next_s@A_mat@next_s+next_s@B_mat-(state@A_mat@state+state@B_mat))-c+delta <= eta)
    #for elem in unsafe_dom_data:
    #    constraints.append(-(elem@A_mat@elem+elem@ B_mat+C_mat)+lambd <= eta)
    #for elem, next_s in zip(state_data, next_states):
    #    constraints.append((next_s@A_mat@next_s+next_s@B_mat-(elem@A_mat@elem+elem@B_mat))/tau-c+delta <= eta)
    #constraints.append(cp.diag(init_dom_data@A_mat@init_dom_data.T)+(init_dom_data@B_mat).flatten()+C_mat-gamma <= eta)
    #constraints.append(-(cp.diag(unsafe_dom_data@A_mat@unsafe_dom_data.T)+(unsafe_dom_data@B_mat).flatten()+C_mat)+lambd <= eta)
    #
    #constraints.append(-(cp.diag(state_data@A_mat@state_data.T)+(state_data@B_mat).flatten() 
    #                    - cp.diag(next_states@A_mat@next_states.T)-(next_states@B_mat).flatten())/tau-c+delta <= eta)

    print("Problem formulated, solving...")
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("eta: {:.5f}".format(eta.value))
    print("matrix values:")
    print(B_mat.value[0])
    print(A_mat.value[0,1]*2)
    print(B_mat.value[1])
    print(C_mat.value)
    print("gamma: {:.5f}".format(gamma.value))
    
    print("Less than zero check: {:.5f}".format(eta.value[0]+L_g*np.sqrt((3.24/np.pi)*eps)))
    cert = certificate(A_mat.value, B_mat.value, C_mat.value)
    if discrete:
        opts = ScenAppConfig(
            SYSTEM=system,
            CERTIFICATE=cert,
            TIME_DOMAIN=TimeDomain.DISCRETE,
        )
    else:
        opts = ScenAppConfig(
            SYSTEM=system,
            CERTIFICATE=cert,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
        )
    axes = plotting.benchmark(system(), cert, levels=[[lambd, gamma.value[0]] ], xrange=[0.1,1], yrange=[0.1,1])
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)

def High_D_test(N):
    max_rhoP = 1.2
    T=2
    mu = -0.
    tau = 1e-15
    
    XD = domains.Rectangle([-2] * 4, [2] * 4)
    XI = domains.Rectangle([0.75, 1.5, 1.5, 1.5], [1, 2, 2, 2])
    from experiments.scenapp_tests.benchmarks.Barr4D import UnsafeDomain
    XU = UnsafeDomain()

    n_data = N 
    

    system = models.Barr4D
    
    
    alpha = 0.01
    N_overline = 1000
    M = 1000
    M=20
    psi = []
    M_f = 0
    for j in tqdm(range(M)):
        lipschitz_data = XD._generate_data(N_overline)()
         
        max_s = 0
        for i, x in enumerate(lipschitz_data):
            y = domains.Sphere(x, alpha)._generate_data(1)()
            inits = torch.vstack((x,y))
            data_tau = system().generate_trajs(inits, tau)[1]
            x_tau = data_tau[0][:,-1]
            y_tau = data_tau[1][:,-1]
            #y_tau = system().generate_trajs(y, tau)[1][0][:,-1]
            x = x.detach().numpy()
            y = y.detach().numpy()
            s = np.linalg.norm((x_tau-x-y_tau+y)/tau)/np.linalg.norm(x-y)
            max_s = max(s,max_s)
            M_f = max(M_f, np.linalg.norm(system().f_torch(0,x))) 
        psi.append(-max_s)
    _, _, L_f, _ = stats.exponweib.fit(psi)
    L_f = -L_f
    

    v = 4
    L_g = 2*max_rhoP*(M_f+v*L_f)
    max_A = np.array([[0.4,.2,.2,.2],[.2,.4,.2,.2],[.2,.2,.4,.2],[.2,.2,.2,.4]]) 
    max_B = np.ones((4,1))*.4
    max_vec = np.ones((4,1))*2
    M_B = max_vec.T@max_A@max_vec+max_B.T@max_vec+0.4
    L_B = np.linalg.norm(2*max_A@max_vec+max_B)
    L = M_B*L_f+M_f*L_B
    delta = 0.5*tau*L*M_f

    lambd = 3.1
    beta = 1e-1
    k = 10+1 #q1-4, gamma, don't count eta
    #k = 5 #

    init_dom_data = XI._generate_data(500)()
    unsafe_dom_data = XU._generate_data(500)()

    init_data = XD._generate_data(n_data)()
    
    all_data = system().generate_trajs(init_data, tau)
    state_data = np.vstack([elem[:, 0] for elem in all_data[1]])
    next_states = np.vstack([elem[:, -1] for elem in all_data[1]])
    
    eps = betaincinv(k, N-k+1, 1-beta)
        
    gap = L_g*(((2**13)/(np.pi**2)*eps)**.25)
    print("Optimality gap: {:.3f}".format(gap))
    
    #import pdb; pdb.set_trace()
    c = cp.Variable()
    A_mat = cp.Variable((4,4), symmetric=True)
    B_mat = cp.Variable((4,1))
    C_mat = cp.Variable(1)

    constraints = [ -0.4 <= A_mat[0,0], A_mat[0,0] <= 0.4, -0.2 <= A_mat[0,1], A_mat[0,1] <= 0.2,
                    -0.2 <= A_mat[0,2], A_mat[0,2] <= 0.2, -0.2 <= A_mat[0,3], A_mat[0,3] <= 0.2,
                    -0.4 <= A_mat[1,1], A_mat[1,1] <= 0.4, -0.2 <= A_mat[1,2], A_mat[1,2] <= 0.2,
                    -0.2 <= A_mat[1,3], A_mat[1,3] <= 0.2, -0.4 <= A_mat[2,2], A_mat[2,2] <= 0.4,
                    -0.2 <= A_mat[2,3], A_mat[2,3] <= 0.2, -0.4 <= A_mat[3,3], A_mat[3,3] <= 0.4,
                    -0.4 <= B_mat, B_mat <= 0.4, 
                    -0.4 <= C_mat, C_mat <=4] # they say this is in [-0.4,0.4] but then find it as 2.7288...

    eta = cp.Variable(1)
    objective = cp.Minimize(eta)

    gamma = cp.Variable(1)
    constraints.append(gamma + c*T - lambd - mu <= eta)

    for init, unsafe, state, next_s in zip(init_dom_data, unsafe_dom_data, state_data, next_states):
        constraints.append(init@A_mat@init+init@B_mat+C_mat-gamma<=eta)
        constraints.append(-(unsafe@A_mat@unsafe+unsafe@ B_mat+C_mat)+lambd <= eta)
        constraints.append((next_s@A_mat@next_s+next_s@B_mat-(state@A_mat@state+state@B_mat))/tau-c+delta <= eta)
    #for elem in unsafe_dom_data:
    #    constraints.append(-(elem@A_mat@elem+elem@ B_mat+C_mat)+lambd <= eta)
    #for elem, next_s in zip(state_data, next_states):
    #    constraints.append((next_s@A_mat@next_s+next_s@B_mat-(elem@A_mat@elem+elem@B_mat))/tau-c+delta <= eta)
    #constraints.append(cp.diag(init_dom_data@A_mat@init_dom_data.T)+(init_dom_data@B_mat).flatten()+C_mat-gamma <= eta)
    #constraints.append(-(cp.diag(unsafe_dom_data@A_mat@unsafe_dom_data.T)+(unsafe_dom_data@B_mat).flatten()+C_mat)+lambd <= eta)
    #
    #constraints.append(-(cp.diag(state_data@A_mat@state_data.T)+(state_data@B_mat).flatten() 
    #                    - cp.diag(next_states@A_mat@next_states.T)-(next_states@B_mat).flatten())/tau-c+delta <= eta)

    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("eta: {:.5f}".format(eta.value))
    print("matrix values:")
    print(A_mat.value)
    print(B_mat.value)
    print(C_mat.value)
    print("gamma: {:.5f}".format(gamma.value))
    
    print("Less than zero check: {:.5f}".format(eta.value[0]+gap))
    
    print(eta.value[0]+gap)
    
    eta = eta.value[0]
    eps = ((-eta/L_g)**4)*(np.pi**2)/(2**13)
    beta = betainc(k, N-k+1, eps)
    print("Maximum confidence level: {:.30f}".format(beta))

    cert = certificate(A_mat.value, B_mat.value, C_mat.value)
    opts = ScenAppConfig(
        SYSTEM=system,
        CERTIFICATE=cert,
    )
    
    eps = ((-eta/L_g)**4)*(np.pi**2)/(2**13)
    N = int(1e19)
    beta = betainc(k, N-k+1, eps)
    print("Maximum confidence level for 1e19 samples (assuming same eta): {:.30f}".format(beta))

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
        
