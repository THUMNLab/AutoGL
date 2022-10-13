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
-----------