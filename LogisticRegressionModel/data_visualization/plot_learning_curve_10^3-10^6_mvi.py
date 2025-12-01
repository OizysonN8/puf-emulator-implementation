import pandas as pd
import matplotlib.pyplot as plt

csv_files = {
    "10^3 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^3_mvi_crp.csv",
    "10^4 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^4_mvi_crp.csv",
    "10^5 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^5_mvi_crp.csv",
    "10^6 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^6_mvi_crp.csv",
}

plt.figure(figsize=(10, 6))

for label, file in csv_files.items():
    df = pd.read_csv(file)

    logreg_df = df[df['model'] == 'logreg']

    x = logreg_df['train_frac']
    y = logreg_df['mean_accuracy']
    yerr = logreg_df['std_accuracy']

    # Plot with error bars
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker='o',
        capsize=3,
        elinewidth=1.2,
        linewidth=2,
        label=label
    )

# Formatting
plt.title("MVI Model Accuracy vs. CRP Dataset Size")
plt.xlabel("Training Fraction")
plt.ylabel("Mean Accuracy")
plt.legend(title="Dataset Size")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig("learning_curve_10^3-10^6_mvi.png", dpi=300)
plt.show()
