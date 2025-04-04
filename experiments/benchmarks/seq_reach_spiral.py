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
import torch
import numpy as np
from experiments.scenapp_tests.benchmarks import models


def test_lnn():
    n_data = 1000
    system = models.Spiral 
    system.time_horizon = 200
    system.T1 = 100
    system.T2 = 100
    #XD = fossil.domains.Sphere([0,0], 1)
    XD1 = domains.Rectangle([-5, 0], [5, 5])
    XD2 = domains.Rectangle([-5, -5], [5, 0])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XG2 = domains.Sphere([0,0],1)
    XG1 = domains.Rectangle([-4, 0], [-1.4, 1])

    SD1 =XD1
    SD2 = domains.Union(domains.SetMinus(XD2, XG2), XG1)
    # Need to have XD does not contain XG (at least for data generation) otherwise might have conflicting requirements on states
    dom = {fossil.XD1: XD1,
            fossil.XD2: XD2,
            fossil.XG1: XG1,
            fossil.XG2: XG2,
            fossil.XG1_BORDER: XG1,
            fossil.XG2_BORDER: XG2,
            fossil.XI: XI
                }
    init_data = XI._generate_data(n_data)()

    all_data = system().generate_trajs(init_data)
    
   
    data = {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}
    
    n_state_data = 1000

    state_data = {fossil.XD1: SD1._generate_data(n_state_data)(),
                  fossil.XD2: SD2._generate_data(n_state_data)(),  
                  fossil.XI: XI._generate_data(n_state_data)(), 
                  fossil.XG1: XG1._generate_data(n_state_data)(),
                  fossil.XG1_BORDER: XG1._sample_border(n_state_data)(),
                  fossil.XG2: XG2._generate_data(n_state_data)(),
                  fossil.XG2_BORDER: XG2._sample_border(n_state_data)()}
    data = {"states_only": state_data, "full_data":data}
    # define NN parameters
    #activations = [fossil.ActivationType.SQUARE]
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [5] * len(activations)

    ###
    #
    ###
    opts = fossil.ScenAppConfig(
        SYSTEM=system,
        DOMAINS=dom,
        N_DATA=n_data,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=fossil.CertificateType.SEQUENTIALREACH,
        TIME_DOMAIN=fossil.TimeDomain.DISCRETE,
        #VERIFIER=fossil.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SCENAPP_MAX_ITERS=200, 
        #LLO=True,
        VERBOSE=2,
    )
    result = fossil.synthesise(opts)

    axes = plotting.benchmark(
        system(), result.cert, domains=opts.DOMAINS, xrange=[-3, 3], yrange=[-3, 3], levels=[[result.cert(state_data[fossil.XG_BORDER]).min().item()]]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
