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
    system.time_horizon = 100
    #XD = fossil.domains.Sphere([0,0], 1)
    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XG = domains.Rectangle([-4, 0], [-1, 0.5])

    SD =domains.SetMinus(XD, XG) 
    dom = {fossil.XD: XD,
            fossil.XG: XG,
            fossil.XG_BORDER: XG,
            fossil.XS_BORDER: XD,
            fossil.XI: XI
                }
    init_data = XI._generate_data(n_data)()

    all_data = system().generate_trajs(init_data)
    
    times = all_data[0]
    states = all_data[1]
    derivs = all_data[2]
    data = {"times":times,"states":states,"derivs":derivs}
    
    n_state_data = 1000

    state_data = {fossil.XD: SD._generate_data(n_state_data)(),
                  fossil.XI: XI._generate_data(n_state_data)(), 
                  fossil.XG: XG._generate_data(n_state_data)(),
                  fossil.XG_BORDER: XG._sample_border(n_state_data)(),
                  fossil.XS_BORDER: XD._sample_border(n_state_data)()}
    data = {"states_only": state_data, "full_data":data}
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [10] * len(activations)

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
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SCENAPP_MAX_ITERS=2,
        VERBOSE=2,
    )
    # keep getting a sub level set not in X_G for some reason??
    result = fossil.synthesise(opts)

    axes = plotting.benchmark(
        system(), result.cert, domains=opts.DOMAINS, xrange=[-3, 3], yrange=[-3, 3]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
