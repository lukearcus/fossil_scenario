from Nejati23.main import Jet_engine, High_D_test
import time
from multiprocessing import Pool
#High_D_test(int(1e19)) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
#Jet_engine(257149)
def solve(ind):
    init = time.perf_counter()
    High_D_test(int(300000), True) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
    post = time.perf_counter()
    print("Elapsed time: {:.5f}".format(post-init))

num_runs = 5
with Pool(processes=num_runs) as pool:
    pool.map(solve, list(range(num_runs)))
