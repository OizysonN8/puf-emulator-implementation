## This file evaluates the bias, reliability, and uniqueness of the Interpose PUF

from pypuf.metrics import bias, reliability, uniqueness
from pypuf.simulation import InterposePUF
from numpy import average

# Create an Interpose PUF
# n=64: Challenge length (number of bits)
# k_up=8: Number of parallel XOR arbiter PUFs in the upper layer
# k_down=8: Number of parallel XOR arbiter PUFs in the lower layer
# seed=1: Random seed for reproducibility
# noisiness=.05: Response noise level
puf = InterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=.00)

# Evaluates bias with traditional 0-1 scale
puf_bias = bias(puf, N=1000, seed=2)
trad_bias = 0.5 - puf_bias/2
print(f'Interpose PUF Bias: {trad_bias:.4f}')

# Evaluate reliability
puf_reliability = average(reliability(puf, N=10000, seed=2), axis=0)[0]
print(f'Interpose PUF Reliability: {puf_reliability:.4f}')

# Uniqueness setup
pufs = [InterposePUF(n=64, k_up=8, k_down=8, seed=i, noisiness=.05) for i in range(10)]

# Evaluate uniqueness
puf_uniqueness = uniqueness(pufs, N=10000, seed=2)[0]
print(f'Interpose PUF Uniqueness: {puf_uniqueness:.4f}')