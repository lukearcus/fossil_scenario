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

    system = models.Spiral
    system.time_horizon = 2500

    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 5])
    SU = domains.Rectangle([4,-1],[5,1])
    
    XG = domains.Sphere([0,0],0.5)

    # Need to have XD does not contain XG (at least for data generation) otherwise might have conflicting requirements on states????
    #SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    XS = domains.SetMinus(XD, SU)
    SD = domains.SetMinus(XS, XG)  # Data for lie set

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XS_BORDER: XS,
        certificate.XS: XS,
        certificate.XG: XG,
    }
    n_data = 1000
    n_state_data = 10000

    # not sure if we should generate data from border of XS? Should be possible for simple borders
    state_data = {
        certificate.XD: XD._generate_data(n_state_data)(),
        certificate.XI: XI._generate_data(n_state_data)(),
        certificate.XS_BORDER: SU._sample_border(n_state_data)(),
        certificate.XG: XG._generate_data(n_state_data)(),
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
    not_goal_inds = [torch.where(domains.Complement(XG).check_containment(torch.Tensor(elem.T))) for elem in all_data[1]]
    times = [torch.Tensor(elem)[inds[0]] for elem, inds in zip(all_data[0], not_goal_inds)]
    states = [elem[:,inds[0]] if len(inds[0]) > 1 else elem[:,inds[0],np.newaxis] if len(inds[0]) == 1 else np.empty([2,0]) for elem, inds in zip(all_data[1], not_goal_inds) ]
    derivs = [elem[:,inds[0]] if len(inds[0]) > 1 else elem[:,inds[0],np.newaxis] if len(inds[0]) == 1 else np.empty([2,0]) for elem, inds in zip(all_data[2], not_goal_inds) ]
    #derivs = [elem[:,inds[0]] for elem, inds in zip(all_data[2], not_goal_inds)]
    # sometimes we end up selecting a single state and get a 1D array...
    data = {"states_only": state_data, "full_data": {"times":times,"states":states,"derivs":derivs}}
    # define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    n_hidden_neurons = [5] * len(activations)

    opts = ScenAppConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=200,
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
