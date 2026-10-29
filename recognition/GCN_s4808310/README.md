
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


## 3. Dataset and Preprocessing

### 3.1 Dataset Description

The project uses the Facebook Large Page-Page Network dataset from the UCI Machine Learning Repository [3]. Each node represents a Facebook page, while an edge represents a mutual like relationship between two pages.

The dataset contains four page categories:

- Company
- Government
- Politician
- TV show

The graph contains 22,470 nodes and 171,002 undirected edges. Each page also has a set of descriptive features, which are used alongside graph connectivity to predict its category.

### 3.2 Feature Preprocessing

The dataset is loaded from three files:

- `musae_facebook_edges.csv`: Connections between Facebook pages.
- `musae_facebook_target.csv`: Node IDs and their classification labels.
- `musae_facebook_features.json`: Feature identifiers associated with each node.

The original node IDs are mapped to consecutive tensor indices, ensuring that the feature matrix, labels and graph edges refer to the same nodes.

A binary feature matrix is constructed with 4,714 columns. Each entry is set to 1 when a page contains the corresponding feature and 0 otherwise.

This produces a node-feature matrix of shape:

`[22470, 4714]`

The four page categories are encoded as integer labels from 0 to 3.

The assignment specification describes a 128-dimensional feature representation. However, this implementation uses the 4,714-dimensional raw features available in the UCI dataset, as identified in the feasibility review.

No dimensionality reduction or feature normalisation is applied.

### 3.3 Graph Construction

The dataset represents an undirected Facebook network.

After mapping the original node identifiers to tensor indices, each edge is stored in both directions so it can be processed by PyTorch Geometric.

The resulting edge tensor has shape:

`[2, 342004]`

The first row contains source indices, while the second contains destination indices.

This bidirectional representation allows information to propagate between connected pages during graph convolution.

### 3.4 Training, Validation and Test Splits

A fixed random seed of 42 is used to generate reproducible node splits.

| Split | Percentage | Nodes |
|---|---:|---:|
| Training | 60% | 13,482 |
| Validation | 20% | 4,494 |
| Testing | 20% | 4,494 |
| **Total** | **100%** | **22,470** |

The splits are represented using Boolean masks:

- `train_mask`
- `val_mask`
- `test_mask`

These masks determine which nodes contribute to training, validation and final testing.

Both the GCN and MLP use the same splits to ensure a fair comparison.

### 3.5 Transductive Learning and Leakage Prevention

The GCN operates in a transductive learning setting.

The complete graph structure and node features are available during training, including the connections of validation and test nodes. However, only training-node labels are used to calculate the training loss.

The training objective is therefore calculated using:

`logits[train_mask]` and `y[train_mask]`

Validation labels are used to monitor performance and select model checkpoints, while test labels are reserved for final evaluation.

This prevents validation and test labels from directly contributing to gradient updates.

The MLP baseline uses the same feature matrix and node splits but does not use graph edges, allowing the effect of neighbourhood information to be compared independently.

### 3.6 Data Verification

The data loader was tested successfully using `dataset.py`.

The verification confirmed:

- 22,470 nodes.
- 342,004 directed edge entries.
- 4,714 input features.
- Four classification categories.
- The expected training, validation and testing node counts.

The graph is returned as a PyTorch Geometric `Data` object containing the node features, edges, labels and split masks.

 
## 4. Model Architecture

Two neural networks are implemented: a Graph Convolutional Network (GCN) and a Multi-Layer Perceptron (MLP). Both use two learnable layers, 64 hidden units, ReLU activation and dropout.

The main difference is that the GCN incorporates graph connectivity, while the MLP classifies each node using only its own features.

### 4.1 Graph Convolutional Network

The GCN follows the graph convolution approach introduced by Kipf and Welling [1].

Unlike a conventional fully connected layer, graph convolution aggregates information from neighbouring nodes. This allows a Facebook page's classification to depend on both its own descriptive features and the characteristics of connected pages.

A graph convolutional layer is defined as:

$$
H^{(l+1)} = \sigma\left(
\tilde{D}^{-1/2}
\tilde{A}
\tilde{D}^{-1/2}
H^{(l)}W^{(l)}
\right)
$$

where:

- $A$ is the graph adjacency matrix.
- $\tilde{A} = A + I$ adds self-loops.
- $\tilde{D}$ is the degree matrix of $\tilde{A}$.
- $H^{(l)}$ contains node representations at layer $l$.
- $W^{(l)}$ is the learnable weight matrix.
- $\sigma$ is the activation function.

The implementation uses two `GCNConv` layers from PyTorch Geometric.

The first layer transforms the 4,714-dimensional input features into 64 hidden features. ReLU activation and dropout are applied before the second layer produces four class scores.

The two-layer architecture allows each node to incorporate information from its local graph neighbourhood, potentially including nodes up to two connections away.

### 4.2 MLP Baseline

A two-layer Multi-Layer Perceptron provides a feature-only baseline.

The MLP applies two fully connected linear transformations with ReLU activation and dropout between them:

$$
H = \operatorname{Dropout}(\operatorname{ReLU}(XW_1+b_1))
$$

$$
Z = HW_2+b_2
$$

The first layer maps 4,714 input features to 64 hidden units, while the second produces four class scores.

Unlike the GCN, the MLP does not use `edge_index` or aggregate information from neighbouring nodes.

This provides a direct comparison between feature-only classification and graph-based classification.

### 4.3 Architecture Comparison

| Property | MLP | GCN |
|---|---|---|
| Input features | 4,714 | 4,714 |
| Hidden units | 64 | 64 |
| Output classes | 4 | 4 |
| Learnable layers | 2 linear | 2 graph convolution |
| Activation | ReLU | ReLU |
| Dropout | 0.5 | 0.5 |
| Graph connectivity | Not used | Used |
| Parameters | 302,020 | 302,020 |

Both models have identical parameter counts, making the comparison particularly useful for assessing the contribution of graph structure.

### 4.4 Classification Objective

Both models produce four raw output scores, known as logits, for each node.

During training, cross-entropy loss is used to compare these scores against the true training-node labels.

The multiclass cross-entropy objective is:

$$
\mathcal{L} =
-\frac{1}{N_{\mathrm{train}}}
\sum_{i \in \mathcal{T}}
\log
\left(
\frac{\exp(z_{i,y_i})}
{\sum_{c=1}^{4}\exp(z_{i,c})}
\right)
$$

where $\mathcal{T}$ is the training-node set and $z_{i,c}$ is the logit for class $c$ at node $i$.

Only training-node labels contribute to this loss.

During evaluation, softmax converts logits into predicted class probabilities. The highest probability determines the predicted category and is used as the model's confidence score.

### 4.5 Implementation

Both architectures are implemented in `modules.py` using PyTorch and PyTorch Geometric.

The file defines two model classes:

- `GCN`: Accepts node features and graph edges.
- `MLP`: Accepts node features only.

The model file also includes a small synthetic-graph smoke test that verifies both models produce outputs with the expected shape.

For ten test nodes and four categories, both models successfully return a tensor of shape `[10, 4]`.
