## This file evaluates the bias, reliability, and uniqueness of the Majority Vote PUF

from pypuf.metrics import bias, reliability, uniqueness
from pypuf.simulation import InterposePUF
from MajorityVoteInterposePUF import MajorityVoteInterposePUF
from numpy import average

# Create an MajorityVoteInterposePUF with an Interpose PUF as the base PUF
# n=64: Challenge length (number of bits)
# k_up=8: Number of parallel XOR arbiter PUFs in the upper layer
# k_down=8: Number of parallel XOR arbiter PUFs in the lower layer
# seed=1: Random seed for reproducibility
# noisiness=.05: Response noise level
# N=20: Number of evaluations per challenge
mvi_puf = MajorityVoteInterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=.05, N=50)

# Evaluates bias with traditional 0-1 scale
puf_bias = bias(mvi_puf, N=1000, seed=2)
trad_bias = 0.5 - puf_bias/2
print(f'Majority Vote PUF Bias: {trad_bias:.4f}')

# Evaluate reliability
puf_reliability = average(reliability(mvi_puf, N=10000, seed=2), axis=0)[0]
print(f'Majority Vote PUF Reliability: {puf_reliability:.4f}')

# Uniqueness setup
mvi_pufs = [MajorityVoteInterposePUF(n=64, k_up=8, k_down=8, seed=i, noisiness=.05, N=50) for i in range(10)]

# Evaluate uniqueness
puf_uniqueness = uniqueness(mvi_pufs, N=10000, seed=2)[0]
print(f'Majority Vote PUF Uniqueness: {puf_uniqueness:.4f}')