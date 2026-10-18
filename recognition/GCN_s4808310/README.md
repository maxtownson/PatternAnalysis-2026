COMP3710 FEASIBILITY REVIEW
Project: Facebook Page Classification using a Multi-layer GNN
Difficulty: Normal
User Need, Scope and Acceptance Criteria
Develop a classification assistant that categorises Facebook pages as government, company, politician, or TV Show. 
Proposed criteria:
-	Compare GCN against an MLP baseline using accuracy, Macro-F1, precision/recall and AUROC.
-	Evaluation whether the confidence of the prediction correlates with real correctness.
-	Investigate a few representative misclassifications.
-	Produce a t-SNE/UMAP embedding visualisation.
-	Measure parameter count, inference latency and peak memory.
Model Choice and Course Concepts
The proposed GCN follows Kipf and Welling's semi-supervised graph convolution framework [1], combining node features with neighbourhood information. The Facebook Page-Page dataset is described by Rozemberczki et al. [2] and obtained from UCI [3].
Architecture:
-	Two layer Graph Convolutional Network
-	64-unit hidden layer, ReLU activation and dropout
-	A feature only MLP will provide the baseline.
Dataset:
-	Facebook Large Page-Page Network 
-	Contains 22470 nodes, 171002 undirected edges and four classification categories
Relevant concepts for the course:
-	Supervised learning, neural networks, graph convolution, cross-entropy optimisation, classification evaluation, uncertainty estimation and data leakage prevention.

 Preliminary Feasibility Evidence
A working PyTorch Geometric data loader has successfully processed the UCI Facebook dataset.
-	22470 nodes and 342004 bidirectional edge entries. 
-	4714 raw binary node features. 
-	Reproducible 60/20/20 train/validation/test masks, containing 13482/4494/4494 nodes respectively. 
-	PyTorch and PyTorch Geometric installed successfully on a Mac with Apple MPS acceleration available.
-	The prescribed PyG dataset download host is inaccessible, so the UCI source was used. It provides raw 4714-dimensional features. So need approval here
Risks, Budget and Fallback
-	Risks: Dataset feature incompatibility, class imbalance, graph label leakage, overconfident predictions and computational limitations.
-	Computing budget: Begin with local CPU/MPS experiments using a compact two-layer GCN; record actual training time and memory usage.
-	Next experiment: Train the MLP and GCN on identical splits, comparing validation macro-F1 and confidence reliability.
-	Fallback: Obtain the prescribed 128-dimensional features from an alternative source or seek approval to use the raw UCI features with documented dimensionality reduction.
References 
[1] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR. https://arxiv.org/abs/1609.02907 
[2] Rozemberczki, B., Allen, C., & Sarkar, R. (2021). Multi-scale attributed node embedding. Journal of Complex Networks, 9(2). https://doi.org/10.1093/comnet/cnab014 
[3] UCI Machine Learning Repository. Facebook Large Page-Page Network. https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network
