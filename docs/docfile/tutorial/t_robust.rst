.. _fe:

Graph Robustness
==========================

Graph robustness is an important research direction in the field of graph representation learning in recent years, 
and we have integrated graph robustness-related algorithms in AutoGL, which can be easily used in conjunction with other modules.

Preliminaries
-----------
In AutoGL, we divide the algorithms for graph robustness into three categories, which are placed in different modules for implementation.
Robust graph feature engineering aims to generate robust graph features in the data pre-processing phase to enhance the robustness of downstream tasks.
Robust graph neural networks, on the other hand, are designed at the model level to ensure the robustness of the model during the training process.
Robust graph neural network architecture search aims to search for a robust graph neural network architecture.
Each of these three types of graph robustness algorithms will be described in the following sections.

Robust Graph Feature Engineering
-----------

Robust Graph Neural Networks
-----------

Robust Graph Neural Architecture Search
---------------------------------------
Robust Graph Neural Architecture Search aims to search for adversarial robust Graph Neural Networks under attacks.
In AutoGL, this module is the code realization of G-RNA. 

Specifically, we design a robust search space for the message-passing mechanism by adding the adjacency mask operations into the search space, 
which is inspired by various defensive operators and allows us to search for defensive GNNs. 
Furthermore, we define a robustness metric to guide the search procedure, which helps to filter robust architectures. 
G-RNA allows us to effectively search for optimal robust GNNs and understand GNN robustness from an architectural perspective.


Adjacency Mask Operations
>>>>>>>>>>>>>>>>>>>>>>>>>
Inspired from the success of current defensive approaches, we conclude the properties of operations on graph structure for robustness and 
design representative defensive operators in our search space accordingly.
In this way, we can choose the most appropriate defensive strategies when confronting perturbed graphs. 
To our best knowledge, this is the first time for the search space to be designed with a specific purpose to enhance the robustness of GNNs.
Specifically, we include five mask operations in the search space. 

- Identity keeps the same adjacency matrix as previous layer
- Low Rank Approximation (LRA) reconstructs the adjacency matrix from the top-k components of singular value decomposition.
- Node Feature Similarity (NFS) deletes edges that have small jaccard similarities among node features.
- Neighbor Importance Estimation (NIE) updates mask values with a pruning strategy base on quantifying the relevance among nodes.
- Variable Power Operator (VPO) forms a variable power graph from the original adjacency matrix weighted by the parameters of influence strengths

Measuring Robustnes
>>>>>>>>>>>>>>>>>>>
Intuitively, the performance of a robust GNN should not deteriorate too much when confronting various perturbed
graph data.