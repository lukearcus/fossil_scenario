from fossil.scenapp import ScenApp, Result
from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main
from experiments.benchmarks import models
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
    
    def generate_boundary(self, v):
        x, y, _, _ = v
        return x == -y**2 

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2, -2, -2], [0, 2, 2, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])
    
    def sample_border(self, batch_size):
        points = []
        limits = [[-2, -2, -2, -2], [0, 2, 2, 2]]
        dom = domains.square_init_data(limits, batch_size)
        dom[:,0] = -dom[:,1]**2
        points = dom
        #idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
        #points += dom[idx][:, 0, :]
        return points
    
    def check_containment(self, x):
        if len(x.shape) == 2:
            return x[:,0] + x[:,1]**2 <= 0
        else:
            return x[0] + x[1]**2 <= 0


def test_lnn(args):
    XD = domains.Rectangle([-5] * 4, [5] * 4)
    XI = domains.Rectangle([0.75, 1.5, 1.5, 1.5], [1, 2, 2, 2])
    XG = domains.Intersection(domains.Rectangle([-2]*4,[2]*4), UnsafeDomain())
    

    n_data = 1000
    num_runs = 5

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XG: XG,
        certificate.XG_BORDER: XG,
        certificate.XS_BORDER: XD,
    }
    state_data = {
        certificate.XD: XD._generate_data(10000)(),
        certificate.XI: XI._generate_data(10000)(),
        certificate.XG: XG._generate_data(10000)(),
        certificate.XG_BORDER: XG._sample_border(10000)(),
        certificate.XS_BORDER: XD._sample_border(10000)()}

    system = models.Barr4D_DT_controlled
    def random_control(obj, t, x):
        return np.array([[.1*(np.random.random()-.5)*(system.u_max-system.u_min)+(system.u_min+system.u_max)/2], [.1*(np.random.random()-.5)*(system.u_max-system.u_min)+(system.u_min+system.u_max)/2]])

    system.controller = random_control
    
    init_data = [XI._generate_data(n_data)() for j in range(num_runs)]
    systems = [[system() for i in range(n_data)] for init_datum in init_data] # parameterised systems
    all_data = [[sys.generate_trajs(np.expand_dims(d,0)) for sys, d in zip(system, init_datum)] for system, init_datum in zip(systems, init_data)]
    times = [[datum[0][0] for datum in all_datum] for all_datum in all_data]
    states = [[datum[1][0] for datum in all_datum] for all_datum in all_data]
    derivs = [[datum[2][0] for datum in all_datum] for all_datum in all_data]
    f_vals = [[datum[3][0] for datum in all_datum] for all_datum in all_data]
    g_vals = [[datum[4][0] for datum in all_datum] for all_datum in all_data]
    
    data = [{"states_only": state_data, "full_data": {"times":time,"states":state,"derivs":deriv, "f_vals":f_val, "g_vals":g_val}} for time, state, deriv, f_val, g_val in zip(times, states, derivs, f_vals, g_vals)]
    #all_data = [system().generate_trajs(init_datum) for init_datum in init_data]

    activations = {"V":[ActivationType.SIGMOID, ActivationType.SIGMOID], "u":[ActivationType.SIGMOID, ActivationType.SIGMOID]}
    
    hidden_neurons = {"V":[25] * len(activations["V"]), "u":[25] * len(activations["u"])}
    opts = [ScenAppConfig(
        N_VARS=4,
        CONTROL_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_data,
        CERTIFICATE=CertificateType.DIRECTCONTROL,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=2500,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        #CONVEX_NET=True,
    ) for datum, system in zip(data,systems)]

    res = [solve(opts[0])]
    import pdb; pdb.set_trace()
    #with Pool(processes=num_runs) as pool:
    #    res = pool.map(solve, opts)
    
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
