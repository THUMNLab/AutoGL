.. _robust:

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

We provide structure engineering methods to enhance robustness, please refer to `preprocessing` part for more information.

Robust Model
-----------

We provides a series of defense methods that aim to enhance the robustness of GNNs.

Building GNNGuard Module
>>>>>>>>>>>>>>>>>>>

Firstly, load pre-attacked graph data:

.. code-block:: python

    perturbed_data = PrePtbDataset(root='/tmp/', name=dataset,attack_method='meta', ptb_rate=0.2)
    modified_adj = perturbed_data.adj

Secondly, train a victim model (GCN) on clearn/poinsed graph:

.. code-block:: python

    flag = False
    print('=== testing GNN on original(clean) graph (AutoGL) ===')
    print("acc_test:",test_autogl(adj, features, device, attention=flag))
    print('=== testing GCN on perturbed graph (AutoGL) ===')
    print("acc_test:",test_autogl(modified_adj, features, device, attention=flag))

For details in training GNN models:

.. code-block:: python

    def test_autogl(adj, features, device, attention):
    ''
    """test on GCN """
    """model_name could be 'GCN', 'GAT', 'GIN','JK'  """
    accs = []
    for seed in tqdm(range(5)):
        # defense model
        gcn = AutoGNNGuard(
                    num_features=pyg_data.num_node_features,
                    num_classes=pyg_data.num_classes,
                    device=args.device,
                    init=False
                ).from_hyper_parameter(model_hp).model
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val=idx_val,
                idx_test=idx_test,
                attention=attention, verbose=True, train_iters=81)
        gcn.eval()
        acc_test, output = gcn.test(idx_test=idx_test)
        accs.append(acc_test.item())
    mean = np.mean(accs)
    std = np.std(accs)
    return {"mean": mean, "std": std}

Thirdly, train defense model GNNGuard on poinsed graph:

.. code-block:: python

    flag = True
    print('=== testing GNN on original(clean) graph (AutoGL) + GNNGuard ===')
    print("acc_test:",test_autogl(adj, features, device, attention=flag))
    print('=== testing GCN on perturbed graph (AutoGL) + GNNGuard ===')
    print("acc_test:",test_autogl(modified_adj, features, device, attention=flag))


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
Inspired by the success of current defensive approaches, we conclude the properties of operations on graph structure for robustness and 
design representative defensive operators in our search space accordingly.
This way, we can choose the most appropriate defensive strategies when confronting perturbed graphs. 
To our knowledge, this is the first time the search space to be designed with a specific purpose to enhance the robustness of GNNs.

Specifically, we include five mask operations in the search space. 

- Identity keeps the same adjacency matrix as previous layer
- Low Rank Approximation (LRA) reconstructs the adjacency matrix from the top-k components of singular value decomposition.
- Node Feature Similarity (NFS) deletes edges that have small jaccard similarities among node features.
- Neighbor Importance Estimation (NIE) updates mask values with a pruning strategy base on quantifying the relevance among nodes.
- Variable Power Operator (VPO) forms a variable power graph from the original adjacency matrix weighted by the parameters of influence strengths.

Measuring Robustness
>>>>>>>>>>>>>>>>>>>>
Intuitively, the performance of a robust GNN should not deteriorate too much when confronting various perturbed
graph data.
we use KL distance to measure the prediction difference between clean and perturbed data.
A larger robustness score indicates a smaller distance between the prediction of clean data and the perturbed data, and consequently, more robust GNN architectures.


Robust Neural Architecture search framework for GNNs: G-RNA
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
G-RNA is able to search for robust Graph Neural Networks based on clean graph data and gain high robustness on perturbed data for searched architectures.

Specifically, G-RNA designs a robust search space for the message-passing mechanism by adding the adjacency matrix mask operations into the search space, 
which comprises various defensive operation candidates and allows us to search for defensive GNNs. 
Furthermore, it defines a robustness metric to guide the search procedure, which helps to filter robust architectures. 
In this way, G-RNA helps understand GNN robustness from an architectural perspective and effectively searches for optimal adversarial robust GNNs.

Here is an example of G-RNA's implementation.

First, set autogl backend and load the dataset.

.. code-block:: python

    # set autogl-backend
    import os
    os.environ["AUTOGL_BACKEND"] = "pyg"

    # load dataset
    from autogl.datasets import build_dataset_from_name
    dataset = build_dataset_from_name('Cora', path='./')

Then, you could define your own GRNA space and GRNA estimator.

.. code-block:: python

    from autogl.module.nas.space import GRNASpace
    from autogl.module.nas.estimator import GRNAEstimator
    from autogl.module.nas.algorithm import GRNA
    space = GRNASpace(
        dropout=0.6,
        input_dim = dataset[0].x.size(1),
        output_dim = dataset[0].y.max().item()+1,
        ops = ['gcn', "gat_2"],
        rob_ops = ["identity","svd","jaccard","gnnguard"],  # graph structure mask operation
        act_ops = ['relu','elu','leaky_relu','tanh']
    )
    estimator = GRNAEstimator(
        lambda_=0.05, 
        perturb_type='random',
        adv_sample_num=10,  
        dis_type='ce',
        ptbr=0.05
    )
    algorithm = GRNA(
        n_warmup=1000,
        population_size=100, 
        sample_size=50, 
        cycles=5000,
        mutation_prob=0.05,
    )

Or, you could simply use GRNA's default parameters.

.. code-block:: python

    from autogl.solver import AutoNodeClassifier
    solver = AutoNodeClassifier(
        graph_models = (),
        ensemble_module = None,
        hpo_module = None, 
        nas_spaces=['grnaspace'],
        nas_algorithms=['grna'],
        nas_estimators=['grna']
        )

Next, search for best robust architecture.

.. code-block:: python

    device = 'cuda'
    solver.fit(dataset)
    solver.get_leaderboard().show()
    orig_acc = solver.evaluate(metric="acc")
    trainer = solver.graph_model_list[0]
    trainer.device = device

After getting the best architecture, we could evaluate on clean/perturbed graph data.

.. code-block:: python

    def metattack(data):
        print('Meta-attack...')
        adj, features, labels = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes), data.x.numpy(), data.y.numpy()
        idx = np.arange(data.num_nodes)
        idx_train, idx_val, idx_test = idx[data.train_mask], idx[data.val_mask], idx[data.test_mask]
        idx_unlabeled = np.union1d(idx_val, idx_test)
        # Setup Surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        # Setup Attack Model
        model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
        # Attack
        n_perturbations = int(data.edge_index.size(1)/2 * 0.05)
        n_perturbations = 1
        model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations, ll_constraint=False)
        perturbed_adj = model.modified_adj
        perturbed_data = data.clone()
        perturbed_data.edge_index = torch.LongTensor(perturbed_adj.nonzero().T)

        return perturbed_data

    from autogl.solver.utils import set_seed
    def test_from_data(trainer, dataset):
        set_seed(0)
        trainer.train(dataset)
        acc = trainer.evaluate(dataset, mask='test')
        return acc
        
    ## test searched model on clean data
    acc = test_from_data(trainer, dataset)

    ## test searched model on perturbed data
    data = dataset[0].cpu()
    dataset[0] = metattack(data).to(device)
    ptb_acc = test_from_data(trainer, dataset)
