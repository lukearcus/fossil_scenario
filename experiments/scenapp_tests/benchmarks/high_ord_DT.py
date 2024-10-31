
from experiments.scenapp_tests.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


def test_lnn(args):
    batch_size = 3000
    f = models.HighOrd8DT
    n_vars = f.n_vars

    XD = domains.Rectangle([-2.2] * n_vars, [2.2] * n_vars)
    # XD = domains.Sphere([0] * n_vars, 2)
    XI = domains.Rectangle([0.9] * n_vars, [1.1] * n_vars)
    # XI = domains.Sphere([1] * n_vars, 0.1)
    XU = domains.Rectangle([-2.2] * n_vars, [-1.8] * n_vars)
    # XU = domains.Sphere([-2] * n_vars, 0.2)
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        # certificate.XS_BORDER: XS,
        certificate.XU: XU,
        # certificate.XG: XG,
    }
    n_data = 1000
    n_state_data = 500 
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    state_data = {
        certificate.XD: XD._generate_data(n_state_data)(),
        certificate.XI: XI._generate_data(n_state_data)(),
        certificate.XU: XU._generate_data(n_state_data)(),
    }
    init_data = XI._generate_data(n_data)()

    system = f 
    all_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, "full_data": {"times":all_data[0],"states":all_data[1],"derivs":all_data[2]}}

    # define NN parameters
    activations = [ActivationType.LINEAR, ActivationType.SIGMOID]
    n_hidden_neurons = [10] * len(activations)

    opts = ScenAppConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=f,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
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
