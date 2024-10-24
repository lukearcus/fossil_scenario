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
    #XD = fossil.domains.Sphere([0,0], 1)
    XD = fossil.domains.Sphere([0, 0], 3)
    
    XI = domains.Rectangle([-1, 1], [1, 2])
    XG = domains.Sphere([0,0],0.5)
    # Need to have XD does not contain XG (at least for data generation) otherwise might have conflicting requirements on states
    dom = {fossil.XD: XD,
            fossil.XG: XG,
            fossil.XI: XI
                }
    init_data = XI._generate_data(n_data)()

    all_data = system().generate_trajs(init_data)
    
    not_goal_inds = [torch.where(domains.Complement(XG).check_containment(torch.Tensor(elem.T))) for elem in all_data[1]]
   
    times = [torch.Tensor(elem)[inds[0]] for elem, inds in zip(all_data[0], not_goal_inds)]
    
    states = [elem[:,inds[0]] if len(inds[0]) > 1 else elem[:,inds[0],np.newaxis] if len(inds[0]) == 1 else np.empty([2,0]) for elem, inds in zip(all_data[1], not_goal_inds) ]
    derivs = [elem[:,inds[0]] if len(inds[0]) > 1 else elem[:,inds[0],np.newaxis] if len(inds[0]) == 1 else np.empty([2,0]) for elem, inds in zip(all_data[2], not_goal_inds) ]
    data = {"times":times,"states":states,"derivs":derivs}
    
    n_state_data = 1000

    state_data = {fossil.XI: XI._generate_data(n_state_data)(), fossil.XG: XG._generate_data(n_state_data)()}
    data = {"states_only": state_data, "full_data":data}

    # define NN parameters
    #activations = [fossil.ActivationType.SQUARE]
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [32] * len(activations)

    ###
    #
    ###
    opts = fossil.ScenAppConfig(
        SYSTEM=system,
        DOMAINS=dom,
        N_DATA=n_data,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=fossil.CertificateType.PRACTICALLYAPUNOV,
        TIME_DOMAIN=fossil.TimeDomain.DISCRETE,
        #VERIFIER=fossil.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SCENAPP_MAX_ITERS=2, 
        #LLO=True,
        VERBOSE=2,
    )
    result = fossil.synthesise(opts)
    
    axes = plotting.benchmark(
        system(), result.cert, domains=opts.DOMAINS, xrange=[-1, 1], yrange=[-1, 1]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
