import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    script_dir,
    "..",
    "LRResults",
    "learning_curve_summary_10^6_mvi_crp.csv"
)

csv_path = os.path.normpath(csv_path)

df = pd.read_csv(csv_path)

df_dummy = df[df["model"] == "dummy"]
df_logreg = df[df["model"] == "logreg"]

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

# Styling
plt.xlabel("Training Samples (mean_n_train)")
plt.ylabel("Mean Accuracy")
plt.title("Learning Curve for MajorityVoteInterposePUF Logistic Regression – 10^6 Dataset")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("learning_curve_10^6_mvi.png", dpi=300)

plt.show()
