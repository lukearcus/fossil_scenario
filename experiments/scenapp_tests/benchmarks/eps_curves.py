
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
from tqdm import tqdm
import matplotlib.pyplot as plt

def test_lnn(args):
    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XU = domains.Rectangle([-5,-1],[-4.5,1])

    n_data = 1000
    
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    state_data = {
        certificate.XD: XD._generate_data(500)(),
        certificate.XI: XI._generate_data(500)(),
        certificate.XU: XU._generate_data(500)(),
    }
    init_data = XI._generate_data(n_data)()

    system = models.Spiral
    system.time_horizon = 100
    all_data = system().generate_trajs(init_data)

    eps_P2L = []
    eps_post = []
    N_vals = list(range(100,1000))
    for i in tqdm(N_vals):
        data = {"states_only": state_data, "full_data": {"times":all_data[0][:i],"states":all_data[1][:i],"derivs":all_data[2][:i]}}

        activations = [ActivationType.SIGMOID]
        #activations = [ActivationType.RELU]
        hidden_neurons = [5] * len(activations)
        opts = ScenAppConfig(
            N_VARS=2,
            SYSTEM=system,
            DOMAINS=sets,
            DATA=data,
            N_DATA=i,
            N_TEST_DATA=i,
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
        

        PAC = ScenApp(opts)
        result = PAC.solve()
        eps_P2L.append(result.res)

        eps_post.append(result.a_post_res)

    plt.plot(N_vals, eps_P2L, label="Risk calculated using certificate")
    plt.plot(N_vals, eps_post, label="Risk calculated directly")
    plt.title("Risk Curves for Varying N")
    plt.xlabel("N")
    plt.ylabel("Risk")
    plt.legend()
    plot_name = f"risk_comparison_curves.pdf"
    plot_name = "results/" + plot_name
    plt.savefig(plot_name)
    #main.run_benchmark(
    #    opts,
    #    record=args.record,
    #    plot=args.plot,
    #    concurrent=args.concurrent,
    #    repeat=args.repeat,
    #)


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
