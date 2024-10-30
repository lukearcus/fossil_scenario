# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pylint: disable=not-callable
from experiments.scenapp_tests.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


def test_lnn(args):
    ###########################################
    ###
    #############################################
    n_vars = 2
    batch_size = 1000
    n_state_data = 1000

    system = models.Spiral
    system.time_horizon = 100

    
    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    SU = domains.Rectangle([4,-1],[5,1])
    XU = domains.Rectangle([4,-1],[5,1])

    XG = domains.Sphere([0,0],1.0)

    XS = domains.SetMinus(XD, XU)
    SD = domains.SetMinus(XS, XG)  # Data for lie set

    sets = {
        "lie": SD,
        "init": XI,
        "safe_border": XS,
        "safe": XS,
        "goal": XG,
        "goal_border": XG,
        "unsafe":SU
    }
    state_data = {
        "lie": SD._generate_data(n_state_data)(),
        "init": XI._generate_data(n_state_data)(),
        "unsafe": XS._sample_border(n_state_data)(),
        "safe": XS._generate_data(n_state_data)(),  # These are just for the beta search
        "goal_border": XG._sample_border(n_state_data)(),
        "goal": XG._generate_data(n_state_data)(),
    }
    init_data = XI._generate_data(batch_size)()

    all_data = system().generate_trajs(init_data)
    
    #derivs = [elem[:,inds[0]] for elem, inds in zip(all_data[2], not_goal_inds)]
    # sometimes we end up selecting a single state and get a 1D array...
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}

    # define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    n_hidden_neurons = [5] * len(activations)

    opts = ScenAppConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RSWS,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SCENAPP_MAX_ITERS=25,
        VERBOSE=2 
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
