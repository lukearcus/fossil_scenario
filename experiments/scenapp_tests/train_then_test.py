# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main
from fossil import scenapp
import fossil
from experiments.benchmarks import models
from fossil.consts import *

from scipy.stats import beta as betaF

class UnsafeDomain(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        return x + y**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])
    
    def check_containment(self, x):
        return x[:,0] + x[:,1]**2 <= 0

def test_lnn(args):
    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.Rectangle([0, 1], [1, 2])
    XU = UnsafeDomain()

    n_data = 500
    
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

    system = models.Barr1
    all_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}

    activations = [ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    #Training
    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_DATA=n_data,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=2500,
    )
    result = fossil.synthesise(opts)
    # Now do verification
    
    test_data = XI._generate_data(n_data)()

    all_test_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, "full_data": {"times":all_test_data[0],"states":all_test_data[1],"derivs":all_test_data[2]}}
    
    all_B_results = [result.cert(torch.tensor(elem.T, dtype=torch.float32)) for elem in data["full_data"]["states"]]
    sat_constraint_B = [all(elem <= 0) for elem in all_B_results]
    all_B_dot_results = [result.cert.nn_dot(torch.tensor(elem.T, dtype=torch.float32), torch.tensor(elem_dot, dtype=torch.float32).T) for elem, elem_dot in zip(data["full_data"]["states"], data["full_data"]["derivs"])]
    sat_constraint_B_dot = [all(elem <= 0) for elem in all_B_dot_results]
    all_constraints = [B and B_dot for B, B_dot in zip(sat_constraint_B, sat_constraint_B_dot)]
    num_to_remove = n_data - sum(all_constraints)
    k = num_to_remove    
    
    beta_bar = (1e-5)/n_data
    N = n_data
    d = 1
    eps = betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 
    print(eps)
    axes = plotting.benchmark(
        system(), result.cert, domains=opts.DOMAINS, xrange=[-3, 2.5], yrange=[-2, 1]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)

if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
