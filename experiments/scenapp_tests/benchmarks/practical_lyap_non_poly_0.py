# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

# from experiments.benchmarks import models
import fossil
from fossil import plotting
from fossil import domains
import torch
import numpy as np

class NonPoly0(fossil.control.DynamicalModel):
    n_vars = 2
    time_horizon = 250

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
    #XD = fossil.domains.Sphere([0,0], 1)
    XG = fossil.domains.Sphere([0,0], 1)
    XD = fossil.domains.Torus([0, 0], 15, 10)
    # Need to have XD does not contain XG (at least for data generation) otherwise might have conflicting requirements on states
    dom = {fossil.XD: XD,
            fossil.XG: XG
                }
    init_data = XD._generate_data(n_data)()

    all_data = system().generate_trajs(init_data)
    
    not_goal_inds = [torch.where(domains.Complement(XG).check_containment(torch.Tensor(elem.T))) for elem in all_data[1]]
    
    times = [elem[inds[0]] for elem, inds in zip(all_data[0], not_goal_inds)]
    
    states = [elem[:,inds[0]] if len(inds[0]) > 1 else elem[:,inds[0],np.newaxis] if len(inds[0]) == 1 else np.empty([2,0]) for elem, inds in zip(all_data[1], not_goal_inds) ]
    derivs = [elem[:,inds[0]] if len(inds[0]) > 1 else elem[:,inds[0],np.newaxis] if len(inds[0]) == 1 else np.empty([2,0]) for elem, inds in zip(all_data[2], not_goal_inds) ]
    
    data = {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}
    
    n_state_data = 1000

    state_data = {fossil.XD: XD._generate_data(n_state_data)(), fossil.XG: XG._generate_data(n_state_data)()}
    data = {"states_only": state_data, "full_data":data}

    # define NN parameters
    #activations = [fossil.ActivationType.SQUARE]
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [6] * len(activations)

    ###
    #
    ###
    opts = fossil.ScenAppConfig(
        SYSTEM=system,
        DOMAINS=dom,
        N_DATA=n_data,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=fossil.CertificateType.PRACTICALLYAPUNOV,
        TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
        #VERIFIER=fossil.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        #LLO=True,
        VERBOSE=2,
    )
    result = fossil.synthesise(opts)
    
    axes = plotting.benchmark(
        system(), result.cert, domains=opts.DOMAINS, xrange=[-3, 2.5], yrange=[-2, 1]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name)


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
