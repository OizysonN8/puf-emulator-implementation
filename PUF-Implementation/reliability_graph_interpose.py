import numpy as np
import matplotlib.pyplot as plt
from pypuf.simulation import InterposePUF
from pypuf.io import random_inputs
from MajorityVoteInterposePUF.MajorityVoteInterposePUF import MajorityVoteInterposePUF

def toBinary(arr: np.ndarray) -> np.ndarray:
    """Converts a -1 and 1 binary array to a 0 and 1 binary array
    Args:
        arr (numpy.ndarray): binary array of 1s and -1s
    Returns:
        numpy.ndarray: binary array of 0s and 1s
    """
    return (arr == 1).astype(int)

# Create InterposePUF and MajorityVoteInterposePUF instances
i_puf = InterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=0.05)
mvi_puf = MajorityVoteInterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=0.05, N=50)

# Generate N random 64-bit challenge vectors using a set random seed
challenges = random_inputs(n=64, N=1000, seed=1)  # shape: (N, 64), values in {-1, +1}

# Repeated Interpose responses (because they're noisy)
i_all_responses = [toBinary(i_puf.eval(challenges)) for _ in range(1000)]
i_all_responses = np.stack(i_all_responses, axis=0)  # (repeats, num_challenges)

# Only need one reference per challenge from the denoised PUF
mvi_ref = toBinary(mvi_puf.eval(challenges))  # shape: (num_challenges,)

# For each challenge, what's the fraction of Interpose responses NOT matching the MVI result?
i_vs_mvi_ber = np.mean(i_all_responses != mvi_ref, axis=0)

# Plot the histogram
plt.figure(figsize=(7,4))
plt.hist(i_vs_mvi_ber, bins=20, range=(0,1), alpha=0.7,
         color="purple", edgecolor="black", label="Bit Error Rate (BER) per Challenge")
plt.axvline(i_vs_mvi_ber.mean(), color="red", linestyle="--",
            label=f"mean BER = {i_vs_mvi_ber.mean():.4f}")
plt.xlabel("Fraction of InterposePUF responses differing from MVI reference")
plt.ylabel("Count (Challenges)")
plt.title("InterposePUF reliability relative to MVI reference")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean disagreement: {i_vs_mvi_ber.mean():.4f}")