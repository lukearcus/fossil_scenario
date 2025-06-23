from Nejati23.main import Jet_engine, High_D_test
import time
from multiprocessing import Pool
from fossil import domains
from experiments.scenapp_tests.benchmarks import models
#High_D_test(int(1e19)) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
#Jet_engine(257149)
def solve(data):
    init = time.perf_counter()
    High_D_test(data) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
    post = time.perf_counter()
    print("Elapsed time: {:.5f}".format(post-init))

num_runs = 5
N = int(300000)
XD = domains.Rectangle([-2] * 4, [2] * 4)
init_data = [XD._generate_data(N)() for i in range(num_runs)]
system = models.Barr4D
data = [system().generate_trajs(init, 2) for init in init_data]
with Pool(processes=num_runs) as pool:
    pool.map(solve, data)
