import logging
import multiprocessing
import time
from queue import Empty

import torch

from fossil import scenapp

def worker_Q(scenApp_config, id, queue, base_seed=0):
        seed = base_seed + id
        torch.manual_seed(seed)
        # np.random.seed(seed)
        c = scenapp.ScenApp(scenApp_config)
        result = c.solve()
        print(result)
        # Remove the functions & symbolic vars from the result to avoid pickling errors
        if isinstance(result.cert, tuple):
            result.cert[0].clean()
            result.cert[1].clean()
        else:
            result.cert.clean()
        result.f.clean()
        calc_PAC_bounds = result.res
        logging.debug("Worker", id, "Found bound of {:.3f}".format(calc_PAC_bounds))
        result_dict = {}
        # Add the id to the label as a sanity check (ensures returned result is from the correct process)
        result_dict["id"] = id
        result_dict["bound"] = calc_PAC_bound
        result_dict["result" + str(id)] = result

        return result_dict

class ScenAppSupervisorQ:
        """Runs CEGIS in parallel and returns the first result found. Uses a queue to communicate with the workers."""

        def __init__(self, timeout_sec=1800, max_P=1):
            self.loop_timeout_sec = timeout_sec
            self.max_processes = max_P

        def solve(self, scenapp_config) -> scenapp.Result:
            stop = False
            procs = []
            queue = multiprocessing.Manager().Queue()
            base_seed = torch.initial_seed()
            id = 0
            n_res = 0
            start = time.perf_counter()
            while not stop:
                while len(procs) < self.max_processes and not stop:
                    p = multiprocessing.Process(
                                                target=worker_Q, args=(scenapp_config, id, queue, base_seed)
                                                                )
                    p.start()
                    id += 1
                    procs.append(p)
                dead = [not p.is_alive() for p in procs]

                try:
                    res = queue.get(block=False)
                    #if res["success"]:
                    logging.debug("Success: Worker", res["id"])
                    [p.terminate() for p in procs]
                    _id = res["id"]
                    result = res["result" + str(_id)]
                    return result
                    #else:
                    #n_res += 1
                    if n_res == self.max_processes:
                        logging.debug("All workers failed, returning last result")
                        # Return the last result
                        _id = res["id"]
                        result = res["result" + str(_id)]
                        return result

                except Empty:
                    pass
