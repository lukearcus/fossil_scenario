# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks import models
from fossil import certificate
from fossil import domains
from fossil import main
from fossil.consts import *
from fossil import plotting


def test_lnn(args):
    n_vars = 3
    system = models.NonPoly2

    # define domain constraints
    inner_radius = 0.01

    XD = domains.Torus([0.0, 0.0, 0.0], 3, 0.01)

    sets = {
        certificate.XD: XD,
    }

    init_data = XD._generate_data(1000)()

    all_data = system().generate_trajs(init_data)
    data = {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}
    
    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [10] * len(activations)

    opts = ScenAppConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_DATA=1000,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        SCENAPP_MAX_ITERS=25,
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
