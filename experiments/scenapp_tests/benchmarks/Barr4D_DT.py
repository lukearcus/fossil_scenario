from fossil.scenapp import ScenApp, Result
from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import *
from functools import partial
from multiprocessing import Pool


def solve(opts):

    PAC = ScenApp(opts)
    result = PAC.solve()
    return result

class UnsafeDomain(domains.Set):
    dimension = 4 

    def generate_domain(self, v):
        x, y, _, _ = v
        return x + y**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2, -2, -2], [0, 2, 2, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])
    
    def check_containment(self, x):
        return x[:,0] + x[:,1]**2 <= 0

def test_lnn(args):
    XD = domains.Rectangle([-2] * 4, [2] * 4)
    XI = domains.Rectangle([0.75, 1.5, 1.5, 1.5], [1, 2, 2, 2])
    XU = UnsafeDomain()

    n_data = 1000
    num_runs = 5

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    state_data = {
        certificate.XD: XD._generate_data(10000)(),
        certificate.XI: XI._generate_data(10000)(),
        certificate.XU: XU._generate_data(10000)(),
    }
    init_data = [XI._generate_data(n_data)() for i in range(num_runs)]

    system = models.Barr4D_DT
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, "full_data": {"times":all_datum[0],"states":all_datum[1],"derivs":all_datum[2]}} for all_datum in all_data]

    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID ]
    #activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    opts = [ScenAppConfig(
        N_VARS=4,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_data,
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
    ) for datum in data]

    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    
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
