
# Facebook Page Classification Using Graph Neural Networks

**COMP3710 Pattern Recognition | Normal Difficulty**

## 1. Project Overview

This project investigates the use of Graph Neural Networks (GNNs) to classify Facebook pages into four categories: company, government, politician, and TV show.

Unlike traditional classifiers, which rely only on the features of an individual page, Graph Neural Networks can also use information from connected pages. This is particularly useful for social networks, where pages with similar characteristics often share connections.

A two-layer Graph Convolutional Network (GCN) is implemented using PyTorch Geometric and compared against a Multi-Layer Perceptron (MLP) baseline. Both models use the same node features and training, validation, and testing splits, allowing the effect of graph structure on classification performance to be investigated.

### 1.1 Research Question

The central question is whether the GCN's improved classification performance also produces more reliable predictions.

A model may achieve high accuracy while remaining overconfident when it makes mistakes. This becomes important when uncertain predictions need to be referred for human review rather than automatically accepted.

The project therefore investigates:

1. Whether graph connectivity improves classification performance compared with a feature-only baseline.
2. Whether predicted confidence corresponds to actual correctness.
3. Whether confidence thresholds can identify predictions that should be referred for review.
4. Which graph structures and node characteristics contribute to classification failures.
5. Whether the improvement in performance justifies the additional computational cost.

### 1.2 Project Scope

The implementation uses the Facebook Large Page-Page Network dataset, containing 22,470 nodes and four classification categories.

The selected model is a two-layer GCN with 64 hidden units, ReLU activation, and dropout. A feature-only MLP serves as the baseline.

The project evaluates classification accuracy, macro-F1, AUROC, confidence calibration, error cases, learned node embeddings, and computational efficiency.
