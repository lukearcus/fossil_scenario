import torch
from fossil import control
from fossil import domains

import timeit

from fossil import plotting
from fossil import certificate
from fossil import main
from fossil.consts import *

class Barr_inv_pendulum(control.DynamicalModel):
    n_vars = 2
    time_horizon = 2

    def f_torch(self, t, v):
        if len(v.shape) == 1:
            theta, theta_dot = v[0], v[1]
        else:
            theta, theta_dot = v[:, 0], v[:, 1]
        #return [theta_dot, 10*np.sin(theta)]
        return [theta_dot, 10*np.sin(theta)-10*np.sin(theta)-1.5*theta-1.5*theta_dot] # with controller


def test_lnn(args):
    XU = domains.SetMinus(domains.Rectangle([-3,-6],[3,6]), domains.Rectangle([-0.3,-0.6],[0.3,0.6]))
    #XU = domains.Complement(domains.Rectangle([-0.3,0.3],[-0.6,0.6]))
    XD = domains.Rectangle([-0.3,-0.6],[0.3,0.6])
    XI = domains.Rectangle([-0.1,-0.1],[0.1,0.1])

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

    system = Barr_inv_pendulum
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
        VERBOSE=2,
        SCENAPP_MAX_ITERS=1000, # Add decreasing step size?
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
