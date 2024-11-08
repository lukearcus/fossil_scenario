from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import *

def test_lnn(args):
    XD = domains.Rectangle([0.1, 0.1], [1, 1])
    XI = domains.Rectangle([0.1, 0.1], [0.5, 0.5])
    XU = domains.Rectangle([0.7, 0.7], [1, 1])

    n_data = 10000 
    
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

    system = models.JetEngBarr
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
