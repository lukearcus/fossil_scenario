# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

# Timing variant of control_barr_spiral.py: 10 trajectories, N_THREADS=4,
# LEARN_LOOPS=1000, VERBOSE=2, single run (no Pool), SCENAPP_MAX_ITERS=10.

import fossil
from fossil import domains
from fossil.consts import *
from fossil.scenapp import ScenApp, Result
import torch
import numpy as np
from experiments.benchmarks import models
from functools import partial
from time import perf_counter


def solve(system, sets, n_data, activations, hidden_neurons, data):

    opts = ScenAppConfig(
        N_VARS=2,
        CONTROL_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_DATA=n_data,
        N_TEST_DATA=n_data,
        CERTIFICATE=CertificateType.DIRECTCONTROLBARR,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
        SCENAPP_MAX_ITERS=10,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        N_THREADS=4,
        LEARN_LOOPS=1000,
    )

    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


def test_lnn():
    n_data = 10
    system = models.SpiralCont
    system.time_horizon = 100
    XD = domains.Rectangle([-15, -15], [15, 15])
    XI = domains.Rectangle([-1, 4], [1, 4.5])
    XU = domains.Sphere([0,0],1)

    dom = {fossil.XD: XD,
            fossil.XU: XU,
            fossil.XI: XI
                }

    n_state_data = 10000

    state_data = {fossil.XD: XD._generate_data(n_state_data)(),
                  fossil.XI: XI._generate_data(n_state_data)(),
                  fossil.XU: XU._generate_data(n_state_data)(),}

    activations = {"V":[fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID], "u":[fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]}

    n_hidden_neurons = {"V":[25] * len(activations["V"]), "u":[25] * len(activations["u"])}

    def random_control(obj, t, x):
        return .1*(np.random.random(2)-.5)*(system.u_max-system.u_min)+(system.u_min+system.u_max)/2
    system.controller = random_control

    init_data = [XI._generate_data(n_data)() for j in range(1)]
    systems = [[system() for i in range(n_data)] for init_datum in init_data]

    t_gen_start = perf_counter()
    all_data = [[sys.generate_trajs(np.expand_dims(d,0)) for sys, d in zip(system, init_datum)] for system, init_datum in zip(systems, init_data)]
    t_gen = perf_counter() - t_gen_start
    print("Initial generate_trajs ({} trajs): {:.3f}s".format(n_data, t_gen))

    times = [[datum[0][0] for datum in all_datum] for all_datum in all_data]
    states = [[datum[1][0] for datum in all_datum] for all_datum in all_data]
    derivs = [[datum[2][0] for datum in all_datum] for all_datum in all_data]
    f_vals = [[datum[3][0] for datum in all_datum] for all_datum in all_data]
    g_vals = [[datum[4][0] for datum in all_datum] for all_datum in all_data]

    data = [{"states_only": state_data, "full_data": {"times":time,"states":state,"derivs":deriv, "f_vals":f_val, "g_vals":g_val}} for time, state, deriv, f_val, g_val in zip(times, states, derivs, f_vals, g_vals)]

    part_solve = partial(solve, systems[0], dom, n_data, activations, n_hidden_neurons)
    t_solve_start = perf_counter()
    res = [part_solve(data[0])]
    t_solve = perf_counter() - t_solve_start
    print("solve() total: {:.3f}s".format(t_solve))

    print("Done. Result: {}".format(res[0]))


if __name__ == "__main__":
    test_lnn()
