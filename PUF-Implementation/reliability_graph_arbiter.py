import numpy as np
import matplotlib.pyplot as plt
from pypuf.simulation import ArbiterPUF
from pypuf.io import random_inputs
from MajorityVoteArbiterPUF.MajorityVoteArbiterPUF import MajorityVoteArbiterPUF

def toBinary(arr: np.ndarray) -> np.ndarray:
    """Converts a -1 and 1 binary array to a 0 and 1 binary array
    Args:
        arr (numpy.ndarray): binary array of 1s and -1s
    Returns:
        numpy.ndarray: binary array of 0s and 1s
    """
    return (arr == 1).astype(int)

# Create ArbiterPUF and MajorityVoteArbiterPUF instances
a_puf = ArbiterPUF(n=64, seed=1, noisiness=0.05)
mva_puf = MajorityVoteArbiterPUF(n=64, seed=1, noisiness=0.05, N=50)

# Generate N random 64-bit challenge vectors using a set random seed
challenges = random_inputs(n=64, N=1000, seed=1)  # shape: (N, 64), values in {-1, +1}

# Repeated ArbiterPUF responses (because they're noisy)
a_all_responses = [toBinary(a_puf.eval(challenges)) for _ in range(1000)]
a_all_responses = np.stack(a_all_responses, axis=0)  # (repeats, num_challenges)

# Only need one reference per challenge from the denoised PUF
mva_ref = toBinary(mva_puf.eval(challenges))  # shape: (num_challenges,)

# For each challenge, what's the fraction of Arbiter responses NOT matching the MVA result?
arbit_vs_mva_ber = np.mean(a_all_responses != mva_ref, axis=0)

# Plot the histogram
plt.figure(figsize=(7,4))
plt.hist(arbit_vs_mva_ber, bins=20, range=(0,1), alpha=0.7,
         color="purple", edgecolor="black", label="Bit Error Rate (BER) per Challenge")
plt.axvline(arbit_vs_mva_ber.mean(), color="red", linestyle="--",
            label=f"mean BER = {arbit_vs_mva_ber.mean():.4f}")
plt.xlabel("Fraction of ArbiterPUF Responses differing from MVA reference")
plt.ylabel("Count (Challenges)")
plt.title("ArbiterPUF reliability relative to MVA reference")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean disagreement: {arbit_vs_mva_ber.mean():.4f}")