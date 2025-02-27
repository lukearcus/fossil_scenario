
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
    XI = domains.Rectangle([0.0, 1], [1, 2])
    XU = UnsafeDomain()

    n_data = 1000
    num_runs = 5
    
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    n_state_data = 10000
    state_data = {
        certificate.XD: XD._generate_data(n_state_data)(),
        certificate.XI: XI._generate_data(n_state_data)(),
        certificate.XU: XU._generate_data(n_state_data)(),
    }
    init_data = [XI._generate_data(n_data)() for i in range(num_runs)]
    
    system = models.Barr1
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, "full_data": {"times":all_datum[0],"states":all_datum[1],"derivs":all_datum[2]}} for all_datum in all_data]

    activations = [ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_data,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=2500,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
    ) for datum in data]
    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    
    axes = plotting.benchmark(
        system(), res[-1].cert, domains=opts[-1].DOMAINS, xrange=[-2, 2], yrange=[-2, 2]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts[-1], name)

    #PAC = ScenApp(opts)
    #result = PAC.solve()
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
