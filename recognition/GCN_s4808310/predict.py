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


def reliability_analysis(probabilities, labels, name):
    """Measure confidence calibration on held-out test nodes."""

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels)

    # Divide prediction confidence into 10 equal-width bins.
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(confidence, bins) - 1, 0, 9)

    mean_confidence = []
    empirical_accuracy = []
    ece = 0.0

    for i in range(10):
        selected = bin_ids == i
        count = int(selected.sum())

        if count == 0:
            continue

        avg_confidence = float(confidence[selected].mean())
        avg_accuracy = float(correct[selected].mean())

        mean_confidence.append(avg_confidence)
        empirical_accuracy.append(avg_accuracy)

        # Weight each bin's calibration gap by its sample count.
        ece += (count / len(confidence)) * abs(
            avg_accuracy - avg_confidence
        )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray", label="Perfect calibration"
    )
    ax.plot(
        mean_confidence,
        empirical_accuracy,
        marker="o",
        label=name.upper()
    )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"{name.upper()} Reliability (ECE={ece:.4f})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}_reliability.png",
        dpi=150
    )
    plt.close(fig)

    return float(ece)


def rejection_analysis(probabilities, labels, name, threshold=None):
    """Calculate accuracy and coverage for confidence thresholds."""

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels

    rows = []

    for t in np.linspace(0.0, 1.0, 201):
        accepted = confidence >= t

        coverage = float(accepted.mean())
        accepted_accuracy = (
            float(correct[accepted].mean())
            if accepted.any() else None
        )

        rows.append({
            "threshold": float(t),
            "coverage": coverage,
            "accuracy": accepted_accuracy,
        })

    if threshold is None:
        # Choose the highest-coverage validation threshold
        # that achieves at least 97% accepted accuracy.
        eligible = [
            row for row in rows
            if row["accuracy"] is not None
            and row["accuracy"] >= 0.97
            and row["coverage"] > 0
        ]

        if not eligible:
            raise ValueError(
                f"{name}: No validation threshold achieves 97% accuracy"
            )

        threshold = max(eligible, key=lambda row: row["coverage"])[
            "threshold"
        ]

    accepted = confidence >= threshold

    result = {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": (
            float(correct[accepted].mean())
            if accepted.any() else None
        ),
        "referred_nodes": int((~accepted).sum()),
        "total_nodes": len(labels),
    }

    return result, rows


def plot_rejection(rows, name):
    """Plot the trade-off between coverage and accepted accuracy."""

    valid = [r for r in rows if r["accuracy"] is not None]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [r["coverage"] for r in valid],
        [r["accuracy"] for r in valid],
        marker=".",
        markersize=3
    )

    ax.set_xlabel("Coverage (fraction automatically classified)")
    ax.set_ylabel("Accuracy among accepted predictions")
    ax.set_title(f"{name.upper()} Coverage–Accuracy Trade-off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_rejection.png", dpi=150)
    plt.close(fig)



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

        # Analyse confidence only on held-out test nodes.
        test_probabilities = probabilities[
            graph.test_mask
        ].cpu().numpy()

        test_labels = graph.y[
            graph.test_mask
        ].cpu().numpy()

        ece = reliability_analysis(
            test_probabilities, test_labels, name
        )

        print(f"{name.upper()} Expected Calibration Error: {ece:.4f}")

        # Select the confidence threshold on validation nodes.
        val_probabilities = probabilities[
            graph.val_mask
        ].cpu().numpy()

        val_labels = graph.y[
            graph.val_mask
        ].cpu().numpy()

        val_result, _ = rejection_analysis(
            val_probabilities, val_labels, name
        )

        selected_threshold = val_result["threshold"]

        # Apply the fixed validation threshold to test nodes.
        test_result, test_rows = rejection_analysis(
            test_probabilities,
            test_labels,
            name,
            threshold=selected_threshold
        )

        plot_rejection(test_rows, name)

        print(
            f"{name.upper()} threshold={selected_threshold:.3f} | "
            f"coverage={test_result['coverage']:.1%} | "
            f"accepted accuracy={test_result['accepted_accuracy']:.1%} | "
            f"referred={test_result['referred_nodes']}"
        )


if __name__ == "__main__":
    main()
