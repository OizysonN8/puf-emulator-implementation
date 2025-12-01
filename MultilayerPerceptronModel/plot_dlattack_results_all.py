import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("MultilayerPerceptronModel/results/all_results.csv")

# Separate MVA and MVI experiments
mva_df = df[df["dataset"].str.contains("MajorityVoteArbiterPUF", case=False)]
mvi_df = df[df["dataset"].str.contains("MajorityVoteInterposePUF", case=False)]

# Sort by CRP size
mva_df = mva_df.sort_values("n_samples")
mvi_df = mvi_df.sort_values("n_samples")

plt.figure(figsize=(10, 6))

# Plot MVA curve
plt.plot(
    mva_df["n_samples"],
    mva_df["test_accuracy"],
    marker="o",
    linewidth=2,
    label="MVA (DL Attack)"
)

# Plot MVI curve
plt.plot(
    mvi_df["n_samples"],
    mvi_df["test_accuracy"],
    marker="o",
    linewidth=2,
    label="MVI (DL Attack)"
)

# Log scale for clarity
plt.xscale("log")

# Formatting
plt.title("Deep Learning Attack Accuracy vs CRP Dataset Size")
plt.xlabel("CRP Count (log scale)")
plt.ylabel("Test Accuracy")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="PUF Type")
plt.tight_layout()

plt.savefig("dl_attack_accuracy_vs_crpsize.png", dpi=300)
plt.show()