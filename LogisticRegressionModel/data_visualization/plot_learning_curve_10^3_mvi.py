import pandas as pd
import matplotlib.pyplot as plt
import os

# Get directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the CSV path relative to the script
csv_path = os.path.join(
    script_dir,
    "..",
    "LRResults",
    "learning_curve_summary_10^3_mvi_crp.csv"
)

csv_path = os.path.normpath(csv_path)

# Read data
df = pd.read_csv(csv_path)

# Separate the two models
df_dummy = df[df["model"] == "dummy"]
df_logreg = df[df["model"] == "logreg"]

# Create the figure
plt.figure(figsize=(8, 5))

# Dummy baseline
plt.plot(
    df_dummy["mean_n_train"],
    df_dummy["mean_accuracy"],
    marker="o",
    linestyle="--",
    label="Baseline (Dummy)"
)

# Logistic regression learning curve
plt.plot(
    df_logreg["mean_n_train"],
    df_logreg["mean_accuracy"],
    marker="o",
    linestyle="-",
    label="Logistic Regression"
)

# Labels + styling
plt.xlabel("Training Samples (mean_n_train)")
plt.ylabel("Mean Accuracy")
plt.title("Learning Curve for MajorityVoteArbiterPUF Logistic Regression – 10³ Dataset")
plt.grid(True, alpha=0.3)
plt.legend()

# Save to file for your report
plt.tight_layout()
plt.savefig("learning_curve_10^3_mvi.png", dpi=300)

# Or show it interactively
plt.show()
