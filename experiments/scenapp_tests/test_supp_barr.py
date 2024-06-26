
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
from experiments.benchmarks import models
from fossil.consts import *
import fossil

from tqdm import tqdm

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

    n_data = 200
    
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

    #system = models.Barr1
    system = models.Barr1_stoch
    all_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}

    activations = [ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    
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
        VERBOSE=1,
        SCENAPP_MAX_ITERS=250,
    )
    result = fossil.synthesise(opts)
    results = [result]
    for i in tqdm(range(n_data)):
        popped_data = {key: data["full_data"][key].pop(i) for key in data["full_data"]}
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
            VERBOSE=1,
            SCENAPP_MAX_ITERS=250,
        )
        result = fossil.synthesise(opts)
        results.append(result)
        _ = {key: data["full_data"][key].insert(i,popped_data[key]) for key in data["full_data"]}
    all_B_results = [elem.cert(torch.tensor(data["full_data"]["states"][0].T, dtype=torch.float32)) for elem in results ]
    norms = [torch.norm(elem-all_B_results[0]) for elem in all_B_results]
    #check_same_res = [all(elem==all_B_results[0]) for elem in all_B_results] # Assuming first sample is not of support (it could be but the chance is small)
    #total_supps = n_data-sum(check_same_res)
    import pdb; pdb.set_trace()
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
