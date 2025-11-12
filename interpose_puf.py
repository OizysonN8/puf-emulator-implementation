from pypuf.simulation import InterposePUF
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

# Create an Interpose PUF
# n=64: Challenge length (number of bits)
# k_up=8: Number of parallel XOR arbiter PUFs in the upper layer
# k_down=8: Number of parallel XOR arbiter PUFs in the lower layer
# seed=1: Random seed for reproducibility
# noisiness=.05: Response noise level
puf = InterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=.05)

# Generate N random 64-bit challenge vectors using a set random seed
challenges = random_inputs(n=64, N=1000, seed=2)

# Simulate the PUF responses to these challenges
responses = puf.eval(challenges)

## Convert the challenges and responses from -1/1 to 0/1 format
challenges = toBinary(challenges)
responses = toBinary(responses)

## Convert the challenges and respones to ints
## challenges = [int(challenge) for challenge in challenges]
responsesFormatted = [int(response) for response in responses]
challengesFormatted = []
for i in range(len(challenges)):
    challengesFormatted.append(int("".join(map(str, challenges[i]))))

## Writes the data to a CSV file
with open('challenge-response-pairs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Challenge', 'Response']) # Header
    for chal, resp in zip(challengesFormatted, responsesFormatted):
        writer.writerow([chal, resp])