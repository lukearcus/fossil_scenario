# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

# from experiments.benchmarks import models
import torch
import fossil
from fossil import plotting
from tqdm import tqdm

class NonPoly0(fossil.control.DynamicalModel):
    n_vars = 2
    time_horizon = 1

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [-x + x * y, -y]

    def f_smt(self, t, v):
        x, y = v
        return [-x + x * y, -y]


def test_lnn():
    n_data = 1000
    system = NonPoly0
    X = fossil.domains.Torus([0, 0], 1, 0.01)
    domain = {fossil.XD: X}
    init_data = X._generate_data(n_data)()
    all_data = system().generate_trajs(init_data)
    data = {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}
    
    #n_data = 1000


    # define NN parameters
    activations = [fossil.ActivationType.SQUARE]
    n_hidden_neurons = [6] * len(activations)

    ###
    #
    ###
    results = []
    for i in tqdm(range(n_data)):
        popped_data = {key: data[key].pop(i) for key in data}
        opts = fossil.ScenAppConfig(
            SYSTEM=system,
            DOMAINS=domain,
            N_DATA=n_data-1,
            DATA=data,
            N_VARS=system.n_vars,
            CERTIFICATE=fossil.CertificateType.LYAPUNOV,
            TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
            #VERIFIER=fossil.VerifierType.DREAL,
            ACTIVATION=activations,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LLO=True,
            SCENAPP_MAX_ITERS=5,
        )
        result = fossil.synthesise(opts)
        results.append(result)
        _ = {key: data[key].insert(i,popped_data[key]) for key in data}
    all_V_results = [elem.cert(torch.tensor(data["states"][0].T, dtype=torch.float32)) for elem in results ]
    check_same_res = [all(elem==all_V_results[0]) for elem in all_V_results] # Assuming first sample is not of support (it could be but the chance is small)
    total_supps = 1000-sum(check_same_res)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
