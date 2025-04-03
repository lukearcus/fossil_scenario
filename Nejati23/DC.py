from Nejati23.main import *
import time
from multiprocessing import Pool
from fossil import domains
from experiments.scenapp_tests.benchmarks import models
#High_D_test(int(1e19)) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
#Jet_engine(257149)
def solve(data):
    init = time.perf_counter()
    DC_Motor(data) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
    post = time.perf_counter()
    print("Elapsed time: {:.5f}".format(post-init))

num_runs = 5
N = 82821

XD = domains.Rectangle([0.1, 0.1], [0.5, 1])
init_data = [XD._generate_data(N)() for i in range(num_runs)]

system = models.DC_Motor

all_data = [system().generate_trajs(init, 2) for init in init_data]
solve(all_data[0])
with Pool(processes=num_runs) as pool:
    pool.map(solve, all_data)
