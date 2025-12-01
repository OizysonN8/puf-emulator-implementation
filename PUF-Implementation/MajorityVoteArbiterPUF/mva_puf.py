from pypuf.simulation import InterposePUF
from MajorityVoteArbiterPUF import MajorityVoteArbiterPUF
from pypuf.io import random_inputs
import numpy as np
import csv

def toBinary(arr: np.ndarray) -> np.ndarray:
    """Converts a -1 and 1 binary array to a 0 and 1 binary array
    Args:
        arr (numpy.ndarray): binary array of 1s and -1s
    Returns:
        numpy.ndarray: binary array of 0s and 1s
    """
    return (arr == 1).astype(int)

# Create an MajorityVoteInterposePUF with an Interpose PUF as the base PUF
# n=64: Challenge length (number of bits)
# k_up=8: Number of parallel XOR arbiter PUFs in the upper layer
# k_down=8: Number of parallel XOR arbiter PUFs in the lower layer
# seed=1: Random seed for reproducibility
# noisiness=.05: Response noise level
# N=20: Number of evaluations per challenge
mva_puf = MajorityVoteArbiterPUF(n=64, seed=5, noisiness=.05, N=50)

# Generate N random 64-bit challenge vectors using a set random seed
challenges = random_inputs(n=64, N=1000000, seed=6)

# Simulate the PUF responses to these challenges
responses = mva_puf.eval(challenges)

## Convert the challenges and responses from -1/1 to 0/1 format
challenges = toBinary(challenges)
responses = toBinary(responses)

## Convert the challenges and respones to ints
## challenges = [int(challenge) for challenge in challenges]
responsesFormatted = [str(response) for response in responses]
challengesFormatted = []
for i in range(len(challenges)):
    challengesFormatted.append("".join(map(str, challenges[i])))

## Writes the data to a CSV file
with open('10^6_mva_crp.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Challenge', 'Response']) # Header
    for chal, resp in zip(challengesFormatted, responsesFormatted):
        writer.writerow([chal, resp])

'''
11/30 Output:
Arbiter PUF Bias: 0.4980
Arbiter PUF Reliability: 0.7158
Arbiter PUF Uniqueness: 0.9359
'''