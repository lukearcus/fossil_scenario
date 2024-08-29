
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


class UnsafeDomain(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        return 2*x**2 + 4*x + 2 + y <= 0

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

    def plot(self, fig, ax, label=None):
        if self.dimension != 2:
            raise NotImplementedError("Plotting only supported for 2D sets")
        colour, label = "red", "Unsafe" 
        
        y = np.linspace(-2,2,50)
        x = -y**2
        ax.plot(x,y, colour, linewidth=2, label=label)
        return fig, ax

def test_lnn(args):
    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.SetMinus(domains.Rectangle([0.25, -1], [1, 1]), domains.Rectangle([0.25, -0.5],[0.75,0.5]))
    XU = domains.SetMinus(UnsafeDomain(), domains.Sphere([0,-1], 0.5))

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

    system = models.Barr1
    all_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}

    activations = [ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
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
        VERBOSE=2,
        SCENAPP_MAX_ITERS=2500,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        #CONVEX_NET=True,
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
