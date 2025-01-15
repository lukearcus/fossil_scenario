
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
from functools import partial
from multiprocessing import Pool

def solve(system, sets, n_data, activations, hidden_neurons, data):

    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
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
    

    PAC = ScenApp(opts)
    result = PAC.solve()
    return result.res, result.a_post_res

def test_lnn(args):
    XD = domains.Rectangle([-5, -5], [5, 5])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XU = domains.Rectangle([-5,-2],[-4,2])

    n_data = 150
    
    eps_P2L = []
    eps_post = []
    min_samples = 100
    max_samples = n_data
    step = 10
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
    system.time_horizon = 100
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]

    N_vals = list(range(min_samples, max_samples, step))
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    for i in tqdm(N_vals):
        eps_P2L_run = []
        eps_post_run = []
        part_solve = partial(solve, system, sets, i, activations, hidden_neurons)
        data = [{"states_only": state_data, "full_data": {"times":all_datum[0][:i],"states":all_datum[1][:i],"derivs":all_datum[2][:i]}} for all_datum in all_data]
        with Pool(processes=num_runs) as pool:
            res = pool.map(part_solve, data)
        eps_P2L.append([j[0] for j in res])
        eps_post.append([j[1] for j in res])

    fig, ax = plt.subplots()
    x_vals = np.array(range(min_samples, max_samples, step)) 
    
    #plt.plot(N_vals, eps_P2L, label="Risk calculated using certificate")
    #plt.plot(N_vals, eps_post, label="Risk calculated directly")
    
    eps_P2L = np.array(eps_P2L).T

    min_P2L = np.min(eps_P2L, axis=0)
    max_P2L = np.max(eps_P2L, axis=0)
    mean_P2L = np.mean(eps_P2L, axis=0)
    top_err = max_P2L-mean_P2L
    bottom_err = mean_P2L-min_P2L

    all_err = np.vstack((bottom_err, top_err))

    ax.errorbar(x_vals, mean_P2L, yerr=all_err, marker="x", linestyle="--", capsize=4, label="Risk calculated with certificate (Theorem 1)")
    
    eps_post = np.array(eps_post).T
    
    min_post = np.min(eps_post, axis=0)
    max_post = np.max(eps_post, axis=0)
    mean_post = np.mean(eps_post, axis=0)
    top_err = max_post-mean_post
    bottom_err = mean_post-min_post

    all_err = np.vstack((bottom_err, top_err))

    ax.errorbar(x_vals, mean_post, yerr=all_err, marker="x", linestyle="--", capsize=4, label="Risk calculated directly (Proposition 5)")
    
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
