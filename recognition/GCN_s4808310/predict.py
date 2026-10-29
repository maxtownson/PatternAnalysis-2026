
from pathlib import Path

import torch

from dataset import load_facebook
from modules import GCN, MLP


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


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


def main():
    device = torch.device("cpu")
    graph = load_facebook(seed=42).to(device)

    for name in ("mlp", "gcn"):
        model = load_model(name, graph, device)
        predictions, probabilities = predict(model, graph, name)

        test_predictions = predictions[graph.test_mask]
        test_labels = graph.y[graph.test_mask]

        accuracy = (
            (test_predictions == test_labels).float().mean().item()
        )

        print(f"{name.upper()} test accuracy: {accuracy:.4f}")
        print("First five predictions:",
              test_predictions[:5].tolist())


if __name__ == "__main__":
    main()
