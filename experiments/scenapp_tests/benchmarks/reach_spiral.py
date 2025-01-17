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
from experiments.scenapp_tests.benchmarks import models
from functools import partial
from multiprocessing import Pool

def solve(system, sets, n_data, activations, hidden_neurons, data):

    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.PRACTICALLYAPUNOV,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=250,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        #CONVEX_NET=True,
    )
    

    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


def test_lnn():
    n_data = 1000
    system = models.Spiral 
    system.time_horizon = 100
    #XD = fossil.domains.Sphere([0,0], 1)
    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XG = domains.Sphere([0,0],1)

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
    #activations = [fossil.ActivationType.SQUARE]
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [5] * len(activations)
    
    num_runs = 1 

    init_data = [XI._generate_data(n_data)() for j in range(num_runs)]
    
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, "full_data": {"times":all_datum[0],"states":all_datum[1],"derivs":all_datum[2]}} for all_datum in all_data]
    part_solve = partial(solve, system, dom, n_data, activations, n_hidden_neurons)
    res = [part_solve(data[0])]
    #with Pool(processes=num_runs) as pool:
    #    res = pool.map(part_solve, data)
    
    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=dom,
        DATA=data[-1],
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.PRACTICALLYAPUNOV,
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
        system(), res[-1].cert, domains=opts.DOMAINS, xrange=[-5, 5], yrange=[-5, 5]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)

if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
