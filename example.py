# Copyright (c) 2023, Alessandro Abate, Alec Edwards,  Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import fossil
from fossil import plotting
import scipy
import torch

class Barr3(fossil.control.DynamicalModel):
    n_vars = 2
    time_horizon = 2 

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            x, y = v[0], v[1]
        else:
            x, y = v[:, 0], v[:, 1]
        return [y, -x - y + 1 / 3 * x**3]

    def f_smt(self, t, v):
        x, y = v
        return [y, -x - y + 1 / 3 * x**3]

    def init_state_dist(self, N_points):
        init_mean = torch.zeros(2)
        init_cov = torch.eye(2)
        init_dist = scipy.stats.multivariate_normal(init_mean, init_cov).rvs
        data=  init_dist(N_points)
        data = torch.tensor(data)
        return data


def test_lnn():
    system = Barr3
    XD = fossil.domains.Rectangle([-3, -2], [2.5, 1])
    XI = fossil.domains.Union(
        fossil.domains.Sphere([1.5, 0], 0.5),
        fossil.domains.Union(
            fossil.domains.Rectangle([-1.8, -0.1], [-1.2, 0.1]),
            fossil.domains.Rectangle([-1.4, -0.5], [-1.2, 0.1]),
        ),
    )

    XU = fossil.domains.Union(
        fossil.domains.Sphere([-1, -1], 0.4),
        fossil.domains.Union(
            fossil.domains.Rectangle([0.4, 0.1], [0.6, 0.5]),
            fossil.domains.Rectangle([0.4, 0.1], [0.8, 0.3]),
        ),
    )

    sets = {
        fossil.XD: XD,
        fossil.XI: XI,
        fossil.XU: XU,
    }
    #Edit this
    data_points = 1000
    init_data = XI._generate_data(data_points)
    all_data = system().generate_trajs(init_data)
    data = {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}
    # define NN parameters
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [10] * len(activations)

    opts = fossil.ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=fossil.CertificateType.BARRIER,
        TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
        N_DATA=data_points,
        #VERIFIER=fossil.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        SCENAPP_MAX_ITERS=25,
        VERBOSE=0,
        SEED=167,
    )

    result = fossil.synthesise(
        opts,
    )
    D = opts.DOMAINS.pop(fossil.XD)
    plotting.benchmark(
        result.f, result.cert, domains=opts.DOMAINS, xrange=[-3, 2.5], yrange=[-2, 1]
    )


if __name__ == "__main__":
    test_lnn()
