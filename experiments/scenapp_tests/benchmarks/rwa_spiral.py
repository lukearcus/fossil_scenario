# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pylint: disable=not-callable
from experiments.scenapp_tests.benchmarks import models
from fossil import domains
from fossil import plotting
from fossil import certificate
from fossil import main, control
from fossil.consts import *
from functools import partial
from multiprocessing import Pool
from fossil.scenapp import ScenApp, Result

def solve(system, sets, n_data, activations, hidden_neurons, data):
    opts = ScenAppConfig(
        DOMAINS=sets,
        DATA=data,
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        SYSTEM=system,
        N_VARS=2,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=2500,
    )
    PAC = ScenApp(opts)
    result = PAC.solve()
    return result

def test_lnn(args):
    ###########################################
    ###
    #############################################
    n_vars = 2
    batch_size = 1000

    system = models.Spiral
    system.time_horizon = 100

    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    SU = domains.Rectangle([-5,-1],[-4.5,1])
    
    XG = domains.Sphere([0,0],1.0)

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
        certificate.XG_BORDER: XG,
    }
    n_data = 1000
    n_state_data = 10000

    # not sure if we should generate data from border of XS? Should be possible for simple borders
    state_data = {
        certificate.XD: SD._generate_data(n_state_data)(),
        certificate.XI: XI._generate_data(n_state_data)(),
        certificate.XS_BORDER: XS._sample_border(n_state_data)(),
        certificate.XG: XG._generate_data(n_state_data)(),
        certificate.XG_BORDER: XG._sample_border(n_state_data)()
    }
    num_runs = 5
    init_data = [XI._generate_data(n_data)() for i in range(num_runs)]

    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    
    data = [{"states_only": state_data, "full_data": {"times":all_datum[0],"states":all_datum[1],"derivs":all_datum[2]}} for all_datum in all_data]
    # define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    n_hidden_neurons = [5] * len(activations)

    #main.run_benchmark(
    #    opts,
    #    record=args.record,
    #    plot=args.plot,
    #    concurrent=args.concurrent,
    #    repeat=args.repeat,
    #)
    
    part_solve = partial(solve, system, sets, n_data, activations, n_hidden_neurons)
    #res = [part_solve(data[0])]
    with Pool(processes=num_runs) as pool:
        res = pool.map(part_solve, data)
    
    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data[-1],
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.RWS,
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
    args = main.parse_benchmark_args()
    test_lnn(args)
