import pandas as pd
import matplotlib.pyplot as plt

csv_files = {
    "10^3 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^3_mva_crp.csv",
    "10^4 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^4_mva_crp.csv",
    "10^5 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^5_mva_crp.csv",
    "10^6 CRPs": "LogisticRegressionModel/lrResults/learning_curve_summary_10^6_mva_crp.csv",
}

plt.figure(figsize=(10, 6))

for label, file in csv_files.items():
    df = pd.read_csv(file)

    mvi_df = df[df['model'] == 'logreg']

    # x-axis = train_frac
    x = mvi_df['train_frac']

    # y-axis = mean_accuracy
    y = mvi_df['mean_accuracy']

    # Plot each dataset's curve
    plt.plot(x, y, marker='o', linewidth=2, label=label)

# Formatting
plt.title("MVA Model Accuracy vs. CRP Dataset Size")
plt.xlabel("Training Fraction")
plt.ylabel("Mean Accuracy")
plt.legend(title="Dataset Size")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig("learning_curve_10^3-10^6_mva.png", dpi=300)
plt.show()