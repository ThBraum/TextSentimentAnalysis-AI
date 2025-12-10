from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve


PLOTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels=(0, 1),
    save_name: Optional[str] = "confusion_matrix.png",
    show: bool = False,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix")
    if save_name:
        path = PLOTS_DIR / save_name
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved confusion matrix to {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_roc(
    y_true: Iterable[int],
    y_score: Iterable[float],
    save_name: Optional[str] = "roc_curve.png",
    show: bool = False,
):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    if save_name:
        path = PLOTS_DIR / save_name
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved ROC curve to {path}")
    if show:
        plt.show()
    plt.close(fig)
