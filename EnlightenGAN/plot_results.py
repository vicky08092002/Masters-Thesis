# plot_results.py
import matplotlib.pyplot as plt

from EnlightenGAN.verify_face import res_clean, res_dark, res_enh


def plot_rocs(results):
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.2f})")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/roc_comparison.png")
    plt.show()

plot_rocs({
    'Clean': res_clean,
    'Dark': res_dark,
    'Enhanced': res_enh
})
