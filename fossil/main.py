# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable


import timeit
import argparse
import warnings

import torch

from fossil.scenapp import ScenApp, Result
import fossil.plotting as plotting
from fossil import cli
from fossil import analysis
from fossil import consts
from fossil.scenapp_supervisor import ScenAppSupervisorQ

"""Top-level module for running benchmarks, and (eventually) using a CLI"""


N_PROCS = 1
BASE_SEED = 0


def parse_benchmark_args():
    """Utility function to allow basic command line interface for running benchmarks."""
    parser = argparse.ArgumentParser(description="Choose basic benchmark options")
    parser.add_argument("--record", action="store_true", help="Record to csv")
    parser.add_argument("--plot", action="store_true", help="Plot benchmark")
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run multiple seeds in parallel and take first successful result",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat over consecutive seeds",
    )
    args = parser.parse_args()
    return args


def run_benchmark(
    scenapp_options: consts.ScenAppConfig,
    repeat=1,
    record=False,
    plot=False,
    concurrent=False,
    **kwargs,
):
    """Unified interface for running benchmarks with a fixed seed.

    Allows for running benchmarks with different configurations regarding recording, plotting, and concurrency.

    Args:
        scenapp_options (ScenAppConfig): Scenario Approach Loop configuration
        repeat (int, optional): How many times to repeat over consecutive seeds. Defaults to 1.
        record (bool, optional): record to csv. Defaults to False.
        plot (bool, optional): plot benchmark. Defaults to False.
        concurrent (bool, optional): For each attempt, run multiple seeds in parallel and take first successful result. Defaults to False.
    """
    torch.set_num_threads(1)

    for i in range(repeat):
        torch.manual_seed(BASE_SEED + i * N_PROCS)
        start = timeit.default_timer()
        if concurrent:
            PAC = ScenAppSupervisorQ(max_P=N_PROCS)
            result = PAC.solve(scenapp_options)
        else:
            PAC = ScenApp(scenapp_options)
            result = PAC.solve()
        stop = timeit.default_timer()
        T = stop - start
        print("Elapsed Time: {}".format(T))

        if plot:
            if scenapp_options.N_VARS != 2:
                warnings.warn("Plotting is only supported for 2-dimensional problems")
            else:
                axes = plotting.benchmark(
                    scenapp_options.SYSTEM(), result.cert, domains=cegis_options.DOMAINS, **kwargs
                )
                for ax, name in axes:
                    plotting.save_plot_with_tags(ax, scenapp_options, name)

        if record:
            rec = analysis.Recorder()
            rec.record(scenapp_options, result, T)


def synthesise(opts: consts.ScenAppConfig) -> Result:
    """Main entry point for synthesising a certificate and controller.

    Args:
        opts (consts.ScenAppConfig): ScenApp configuration

    Returns:
        Result: Result of synthesis (success flag, certificate, final model and stats)

    """

    torch.set_num_threads(1)
    if opts.SEED is not None:
        torch.manual_seed(opts.SEED)
    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


def learn(opts: consts.ScenAppConfig) -> Result:
    """Learns a certificate and controller without verification step.

    Args:
        opts (consts.ScenAppConfig): ScenApp configuration
    Returns:
        Result: Result of synthesis (success flag, certificate, final model and stats)

    """
    raise NotImplementedError


def verify(opts: consts.ScenAppConfig) -> Result:
    """Verifies a given certificate and controller.

    Args:
        opts (consts.CegisConfig): Cegis configuration
    Returns:
        Result: Result of synthesis (success flag, certificate, final model and stats)

    """
    raise NotImplementedError


def _cli_entry():
    """Main entry point for running benchmarks."""
    args = cli.parse_filename()
    if args.certificate is not None:
        cli.print_certificate_sets(args.certificate)
    opts = cli.parse_yaml_to_cegis_config(args.file)
    result = synthesise(opts)
    if args.plot:
        if opts.N_VARS != 2:
            warnings.warn("Plotting is only supported for 2-dimensional problems")
        else:
            axes = plotting.benchmark(result.f, result.cert, domains=opts.DOMAINS)
            for ax, type in axes:
                plotting.save_plot_with_tags(ax, opts, type)


if __name__ == "__main__":
    _cli_entry()
