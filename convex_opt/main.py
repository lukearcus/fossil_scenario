import numpy as np
import cvxpy as cp

from fossil import domains
from fossil import plotting
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import *
from scipy.special import betaincinv, betainc
from scipy import stats
from tqdm import tqdm

import torch
torch.manual_seed(0)

def Jet_engine(N):
    
    T=2
    XD = domains.Rectangle([0.1, 0.1], [1, 1])
    XI = domains.Rectangle([0.1, 0.1], [0.5, 0.5])
    XU = domains.Rectangle([0.7, 0.7], [1, 1])
    init_dom_data = XI._generate_data(500)()
    unsafe_dom_data = XU._generate_data(500)()
    
    init_data = XI._generate_data(N)()

    system = models.JetEngBarr
    all_data = system().generate_trajs(init_data, T)
    #state_data = np.vstack([elem[:, 0] for elem in all_data[1]])
    #next_states = np.vstack([elem[:, -1] for elem in all_data[1]])
    
    W_1 = cp.Variable((5,2))
    b_1 = cp.Variable((5,1))
    W_2 = cp.Variable((1,5))
    b_2 = cp.Variable(1)

    xi = cp.Variable(len(all_data[0]))
    objective = cp.Minimize(cp.sum(xi))

    import pdb; pdb.set_trace()
    constraints = []
    for i, (state, next_s, time) in enumerate(zip(all_data[1],all_data[2], all_data[0])):
        initial_inds = torch.where(XI.check_containment(torch.Tensor(state.T)))
        unsafe_inds = torch.where(XU.check_containment(torch.Tensor(state.T)))
        
        if len(initial_inds[0])>1:
            constraints.append(W_2@cp.pos(W_1@state[:, initial_inds[0]]+b_1)+b_2<= xi[i])
        elif len(initial_inds[0]) == 1:
            constraints.append(W_2@cp.pos(W_1@state[:, initial_inds[0], np.newaxis]+b_1)+b_2<= xi[i])
        time = np.expand_dims(time,0)
        if len(unsafe_inds[0]) > 1:
            constraints.append(-(W_2@cp.pos(W_1@state[:, unsafe_inds[0]]+b_1)+b_2)<= xi[i])
        elif len(unsafe_inds[0]) == 1:
            constraints.append(-(W_2@cp.pos(W_1@state[:, unsafe_inds[0], np.newaxis]+b_1)+b_2)<= xi[i])
        constraints.append((W_2@(cp.pos(W_1@next_s+b_1)-cp.pos(W_1@state+b_1)))/time<= xi[i])
    import pdb; pdb.set_trace()
        

    prob = cp.Problem(objective, constraints)
    prob.solve()
    import pdb; pdb.set_trace()
    print(eta.value)
    print(B_mat.value[0])
    print(A_mat.value[0,1]*2)
    print(B_mat.value[1])
    print(C_mat.value)
    print(gamma.value)
    
    print(eta.value[0]+L_g*np.sqrt((3.24/np.pi)*eps))
    cert = certificate(A_mat.value, B_mat.value, C_mat.value)
    opts = ScenAppConfig(
        SYSTEM=system,
        CERTIFICATE=cert,
    )
    axes = plotting.benchmark(system(), cert, levels=[[lambd, gamma.value[0]] ], xrange=[0.1,1], yrange=[0.1,1])
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)
    


