from Nejati23.main import Jet_engine, High_D_test
import time
#High_D_test(int(1e19)) # run with ~ 100 samples to see what eta we get, then see how many samples we need to keep this negative
init = time.perf_counter()
#Jet_engine(257149)
Jet_engine(1000)
post = time.perf_counter()
print("Elapsed time: {:.5f}".format(post-init))
