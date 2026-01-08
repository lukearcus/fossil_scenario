# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

# from experiments.benchmarks import models
import fossil
from fossil import plotting
from fossil import domains
from fossil.consts import *
from fossil.scenapp import ScenApp, Result
import torch
import numpy as np
from experiments.benchmarks import models
from functools import partial
from multiprocessing import Pool
import torch
torch.set_num_threads(8)
torch.manual_seed(0)

def solve(systems, sets, n_data, activations, hidden_neurons, data):

    opts = ScenAppConfig(
        N_VARS=2,
        CONTROL_VARS=1,
        SYSTEM=systems,
        DOMAINS=sets,
        DATA=data,
        N_DATA=n_data,
        N_TEST_DATA=100,
        CERTIFICATE=CertificateType.DIRECTCONTROL,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=1000,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        #CONVEX_NET=True,
    )
    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


def test_lnn():
    n_data = 30
    system = models.InvPendulum 
    
    def random_control(obj, t, x):
        return .1*(np.random.random()-.5)*(system.u_max-system.u_min)+(system.u_min+system.u_max)/2
    system.controller = random_control

    def uncontrol(obj, t, x):
        return np.array([0])
    
    def stab_control(obj, t, x):
        if type(x) is torch.Tensor:
            x = x.numpy()
        u= -(system.gr / system.l ) * np.sin(x[0])* (3/system.I) - 0.1*(x[0])
        return u
    system.controller = stab_control
    # next: in scenapp, initialise controller nn to be like this one

    #system.controller = uncontrol
    #XD = fossil.domains.Sphere([0,0], 1)
    #XD = domains.Rectangle([-5, -5], [5, 5])
    #XD = domains.Sphere([0,0],5)
    #XI = domains.Torus([0,0],3,2)
    ##XI = domains.Rectangle([-3, -3], [3, 3])
    #XG = domains.Sphere([0,0],0.1)
    
    XD = domains.Rectangle([-5.3, -5.6], [5.3, 5.6])
    #XI = domains.Rectangle([-0.11, -0.31], [-0.09, -0.29])
    XI = domains.Sphere([-0.1,-0.3],0.01)
    XG = domains.Sphere([0,0],0.1) # can we converge to bottom? # back to top, start with stab controller
    XG_enlarged = domains.Sphere([0,0],0.15) # can we converge to bottom? # back to top, start with stab controller

    SD =domains.SetMinus(XD, XG) 
    # Need to have XD does not contain XG (at least for data generation) otherwise might have conflicting requirements on states
    dom = {fossil.XD: XD,
            fossil.XG: XG,
            fossil.XG_BORDER: XG,
            fossil.XS_BORDER: XD,
            fossil.XI: XI
                }
    
    n_state_data = 10000

    state_data = {fossil.XD: SD._generate_data(n_state_data)(),
                  fossil.XI: XI._generate_data(n_state_data)(), 
                  fossil.XG: XG._generate_data(n_state_data)(),
                  fossil.XG_BORDER: XG._sample_border(n_state_data)(),
                  fossil.XS_BORDER: XD._sample_border(n_state_data)()}
    # define NN parameters
    dom = {fossil.XD: XD,
            fossil.XG: XG_enlarged,
            fossil.XG_BORDER: XG,
            fossil.XS_BORDER: XD,
            fossil.XI: XI
                }
    activations = {"V":[fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID], "u":[fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]}
    
    n_hidden_neurons = {"V":[25] * len(activations["V"]), "u":[25] * len(activations["u"])}
    
    num_traj_plots = 5
    init_data = XI._generate_data(num_traj_plots)()
    traj_data_random = system().generate_trajs(init_data)[1]
    
    num_runs =5

    init_data = [XI._generate_data(n_data)() for j in range(num_runs)]
   
    systems = [[system() for i in range(n_data)] for init_datum in init_data] # parameterised systems
    all_data = [[sys.generate_trajs(np.expand_dims(d,0)) for sys, d in zip(system, init_datum)] for system, init_datum in zip(systems, init_data)]
    
    times = [[datum[0][0] for datum in all_datum] for all_datum in all_data]
    states = [[datum[1][0] for datum in all_datum] for all_datum in all_data]
    derivs = [[datum[2][0] for datum in all_datum] for all_datum in all_data]
    f_vals = [[datum[3][0] for datum in all_datum] for all_datum in all_data]
    g_vals = [[datum[4][0] for datum in all_datum] for all_datum in all_data]
    
    data = [{"states_only": state_data, "full_data": {"times":time,"states":state,"derivs":deriv, "f_vals":f_val, "g_vals":g_val}} for time, state, deriv, f_val, g_val in zip(times, states, derivs, f_vals, g_vals)]
    part_solve = partial(solve, systems[0], dom, n_data, activations, n_hidden_neurons) # parallel broken by systems[0]
    res = [part_solve(data[0])]
    #with Pool(processes=num_runs) as pool:
    #    res = pool.map(part_solve, data)
    
    def diss_control(obj, t, x):
        x = torch.tensor(x,dtype=torch.float32)
        if len(x.shape) == 1: 
            return res[-1].cert[1](x.unsqueeze(1).T).detach().numpy()
        else:
            return res[-1].cert[1](x.unsqueeze(2).mT).detach().numpy()
        
    system.controller = diss_control
    
    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=dom,
        DATA=data[-1],
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.DIRECTCONTROL,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=2500,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        #CONVEX_NET=True,
    )
    axes = plotting.benchmark(
        system(), res[-1].cert[0], domains=opts.DOMAINS, xrange=[-5, 5], yrange=[-5, 5]
    )

    init_data = XI._generate_data(num_traj_plots)()
    traj_data_controlled = [system().generate_trajs(np.expand_dims(init_datum,0))[1][0] for init_datum in init_data]
    for traj in traj_data_controlled:
        axes[0][0].plot(traj[0,:], traj[1,:], 'b')
    #for traj in traj_data_random:
    #    axes[0][0].plot(traj[0,:], traj[1,:], 'r')
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
