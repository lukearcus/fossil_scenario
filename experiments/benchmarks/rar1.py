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
    n_vars = 2

    system = models.SecondOrderLQR
    n_state_data = 1000
    n_data = 1000

    XD = domains.Rectangle([-3.5, -3.5], [3.5, 3.5])
    XS = domains.Rectangle([-3, -3], [3, 3])
    XI = domains.Rectangle([-2, -2], [2, 2])
    XG = domains.Rectangle([-0.1, -0.1], [0.1, 0.1])
    XF = domains.Rectangle([-0.15, -0.15], [0.15, 0.15])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    SD = domains.SetMinus(XS, XG)  # Data for lie set
    SNF = domains.SetMinus(XD, XF)

    sets = {
        "lie": XD,
        "init": XI,
        "unsafe": SU,
        "safe_border": XS,
        "safe": XS,
        "goal": XG,
        "final": XF,
        "not_final": SNF
    }
    state_data = {
        "lie": SD._generate_data(n_state_data)(),
        "init": XI._generate_data(n_state_data)(),
        "unsafe": SU._generate_data(n_state_data)(),
        "goal": XG._generate_data(n_state_data)(),
        "final": XF._generate_data(n_state_data)(),
        "not_final": SNF._generate_data(n_state_data)(),
        }
    
    init_data = XI._generate_data(n_data)()

    all_data = system().generate_trajs(init_data)
    
    #derivs = [elem[:,inds[0]] for elem, inds in zip(all_data[2], not_goal_inds)]
    # sometimes we end up selecting a single state and get a 1D array...
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [6] * len(activations)

    activations_alt = [ActivationType.SQUARE]
    n_hidden_neurons_alt = [6] * len(activations_alt)

    opts = ScenAppConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RAR,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        ACTIVATION_ALT=activations_alt,
        N_HIDDEN_NEURONS_ALT=n_hidden_neurons_alt,
        SCENAPP_MAX_ITERS=100,
        SYMMETRIC_BELT=False,
        VERBOSE=2,
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
        # xrange=[-0.2, 0.2],
        # yrange=[-0.2, 0.2],
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
