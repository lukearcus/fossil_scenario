# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

from fossil.scenapp import ScenApp, Result
from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import *
from multiprocessing import Pool

def solve(opts):

    PAC = ScenApp(opts)
    result = PAC.solve()
    return result

def test_lnn(args):
    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XU = domains.Rectangle([-5,-2],[-4,2])

    n_data = 1000
    num_runs = 5

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    n_state_data = 1000
    state_data = {
        certificate.XD: XD._generate_data(n_state_data)(),
        certificate.XI: XI._generate_data(n_state_data)(),
        certificate.XU: XU._generate_data(n_state_data)(),
    }
    init_data = [XI._generate_data(n_data)() for i in range(num_runs)]
    
    system = models.Spiral
    system.time_horizon = 50
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, "full_data": {"times":all_datum[0],"states":all_datum[1],"derivs":all_datum[2]}} for all_datum in all_data]
    
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=2000,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        #CONVEX_NET=True,
    ) for datum in data]
    opts[0].VERBOSE = 2 

    #PAC = ScenApp(opts)
    #result = PAC.solve()
    #main.run_benchmark(
    #    opts,
    #    record=args.record,
    #    plot=args.plot,
    #    concurrent=args.concurrent,
    #    repeat=args.repeat,
    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    #res = [solve(opt) for opt in opts]

    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data[-1],
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
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
        plotting.save_plot_with_tags(ax, opts, name+"_unsafe")


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
