from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
)

from dataset import load_facebook
from modules import GCN, MLP


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = ["company", "government", "politician", "tvshow"]

def load_model(name, graph, device):
    model_class = GCN if name == "gcn" else MLP

    model = model_class(
        input_dim=graph.num_node_features,
        hidden_dim=64,
        num_classes=4
    ).to(device)

    checkpoint = OUTPUT_DIR / f"{name}_best.pt"
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict(model, graph, name):
    with torch.no_grad():
        if name == "gcn":
            logits = model(graph.x, graph.edge_index)
        else:
            logits = model(graph.x)

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return predictions, probabilities

def evaluate_classification(predictions, probabilities, graph, name):
    """Evaluate predictions on held-out test nodes."""

    mask = graph.test_mask

    y_true = graph.y[mask].cpu().numpy()
    y_pred = predictions[mask].cpu().numpy()
    y_prob = probabilities[mask].cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    auroc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=np.arange(4)
    )

    print(f"\n{name.upper()} Classification Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro-AUROC: {auroc:.4f}")

    print(classification_report(
        y_true,
        y_pred,
        labels=range(4),
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    matrix = confusion_matrix(y_true, y_pred, labels=range(4))

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"{name.upper()} Test Confusion Matrix")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_confusion.png", dpi=150)
    plt.close(fig)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "auroc": float(auroc),
    }


def main():
    device = torch.device("cpu")
    graph = load_facebook(seed=42).to(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("mlp", "gcn"):
        model = load_model(name, graph, device)
        predictions, probabilities = predict(model, graph, name)

        evaluate_classification(
            predictions, probabilities, graph, name
        )


if __name__ == "__main__":
    main()
