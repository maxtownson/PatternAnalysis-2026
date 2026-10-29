
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


## 2. Feasibility Review

The following feasibility review was presented during the compulsory project consultation and approved for progression.

### 2.1 User Need, Scope and Acceptance Criteria

The project aims to develop a classification assistant that categorises Facebook pages into four classes: government, company, politician and TV show.

The proposed acceptance criteria are:

1. Compare the GCN against an MLP baseline using accuracy, macro-F1, precision, recall and AUROC.
2. Evaluate whether prediction confidence correlates with actual correctness.
3. Investigate representative misclassifications to identify possible sources of error.
4. Produce a t-SNE or UMAP visualisation of learned node representations.
5. Measure model parameter count, inference latency and memory usage.

### 2.2 Model Choice and Course Concepts

The selected architecture is a two-layer Graph Convolutional Network (GCN), following the semi-supervised graph convolution framework proposed by Kipf and Welling [1].

The GCN combines individual node features with information aggregated from neighbouring nodes. A feature-only Multi-Layer Perceptron (MLP) provides the baseline for evaluating the contribution of graph connectivity.

The proposed architecture consists of:

- Two graph convolution layers.
- A hidden layer containing 64 units.
- ReLU activation and dropout.
- Four output classes corresponding to the Facebook page categories.

The relevant course concepts include supervised learning, neural networks, graph convolution, cross-entropy optimisation, classification evaluation, uncertainty estimation and data leakage prevention.

### 2.3 Preliminary Feasibility Evidence

The preliminary data-loading implementation successfully processed the Facebook Large Page-Page Network dataset obtained from the UCI Machine Learning Repository [3].

The initial data audit established:

| Property | Result |
|---|---|
| Nodes | 22,470 |
| Undirected edges | 171,002 |
| Bidirectional edge entries | 342,004 |
| Raw node features | 4,714 |
| Classification categories | 4 |
| Training nodes | 13,482 |
| Validation nodes | 4,494 |
| Testing nodes | 4,494 |

The train/validation/test split follows a reproducible 60/20/20 allocation using a fixed random seed.

PyTorch and PyTorch Geometric were successfully installed and tested on a Mac with Apple MPS acceleration.

**Dataset compatibility consideration:** The assessment specification describes a 128-dimensional feature representation, whereas the accessible UCI dataset provides 4,714 raw binary features. This difference was identified in the feasibility review as requiring approval or an appropriate fallback. The implemented experiments use the raw feature representation, and this distinction is retained in the final methodology.

### 2.4 Risks, Computing Budget and Fallback

The initial feasibility review identified several potential risks:

- Dataset feature incompatibility between the supplied specification and available UCI data.
- Class imbalance affecting aggregate classification performance.
- Label leakage during graph-based message passing.
- Overconfident predictions on ambiguous or incorrectly classified nodes.
- Computational limitations when training and evaluating graph models.

The proposed computing strategy was to begin with a compact, two-layer GCN using local CPU or Apple MPS resources, then measure actual training time and memory requirements.

The next planned experiment was to train the GCN and MLP on identical data splits and compare their validation macro-F1 and confidence reliability.

If the raw features proved unsuitable, the proposed fallback was to obtain the prescribed 128-dimensional features or seek approval for raw features with documented dimensionality reduction.

### 2.5 Feasibility Review References

[1] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR. https://arxiv.org/abs/1609.02907

[2] Rozemberczki, B., Allen, C., & Sarkar, R. (2021). Multi-scale attributed node embedding. Journal of Complex Networks, 9(2). https://doi.org/10.1093/comnet/cnab014

[3] UCI Machine Learning Repository. Facebook Large Page-Page Network. https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network
