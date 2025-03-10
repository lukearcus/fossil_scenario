from fossil import plotting
from fossil.scenapp import ScenApp, Result
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

def test_lnn(args):
    XD = domains.Rectangle([0.1, 0.1], [1, 1])
    XU = domains.Rectangle([0.1, 0.1], [0.5, 0.5])
    XI = domains.Rectangle([0.7, 0.3], [1, 0.6])

    n_data = 100 
    num_runs = 1

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

    system = models.JetEngBarr
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, "full_data": {"times":all_datum[0],"states":all_datum[1],"derivs":all_datum[2]}} for all_datum in all_data]
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
    hidden_neurons = [10] * len(activations)
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_data,
        BETA=(0.01,),
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=10,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
    ) for datum in data]
    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    
    axes = plotting.benchmark(
        system(), res[-1].cert, domains=opts[-1].DOMAINS, xrange=[0.1, 1], yrange=[0.1, 1]
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts[-1], name)
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
