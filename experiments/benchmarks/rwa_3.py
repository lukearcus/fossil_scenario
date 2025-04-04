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
    system = models.ThirdOrderLQR
    n_vars = 3
    batch_size = 1000

    XD = domains.Rectangle([-6, -6, -6], [6, 6, 6])
    XS = domains.Rectangle([-5, -5, -5], [5, 5, 5])
    XI = domains.Rectangle([-1.2, -1.2, -1.2], [1.2, 1.2, 1.2])
    XG = domains.Rectangle([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    SD = domains.SetMinus(XS, XG)  # Data for lie set
    
    n_data = 1000
    n_state_data = 1000

    state_data = {
        certificate.XI: XI._generate_data(n_state_data)(),
        certificate.XU: SU._generate_data(n_state_data)(),
    }
    init_data = XI._generate_data(n_data)()
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XS_BORDER: XS,
        certificate.XS: XS,
        certificate.XG: XG,
    }

    all_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}
    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [16] * len(activations)

    opts = ScenAppConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
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
