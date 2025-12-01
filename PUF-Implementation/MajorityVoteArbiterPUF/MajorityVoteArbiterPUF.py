'''
MajorityVoteInterposePUF is a class of PUF that solves for reliability issues
in noisy Arbiter PUFs by taking the majority vote of N evaluations of the
base PUF for each challenge.
'''

from pypuf.simulation import ArbiterPUF
import numpy as np

class MajorityVoteArbiterPUF(ArbiterPUF):
    ## Instantiates a Majority Vote PUF
    ## base_puf: The base PUF to be evaluated multiple times
    ## N: Number of evaluations per challenge
    def __init__(self, n: int, seed: int = None,
                 noisiness: float = 0, N: int=20) -> None:
        super().__init__(n=n, seed=seed, noisiness=noisiness)
        self.N = N

    ## Evaluates the Majority Vote PUF on a set of challenges
    ## challenges: 2D numpy array of shape (M, n) where M is the number of challenges
    ##             and n is the challenge length
    def eval(self, challenges: np.ndarray) -> np.ndarray:
        reps = np.array([super().eval(challenges) for _ in range(self.N)])  # shape (N, M), values -1/+1
        sums = np.sum(reps, axis=0)
        # majority in ±1 domain; tie break arbitrarily as +1
        votes = np.where(sums >= 0, 1, -1).astype(np.int8)
        return votes

