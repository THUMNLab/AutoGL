.. _robust_cn:

图鲁棒性
============================

图鲁棒性是近年图机器学习领域重要的研究方向，我们在AutoGL中集成了图鲁棒性相关算法，可以方便地与其他模块结合使用。

背景知识
------------

（介绍对抗攻击、鲁棒问题的定义等，可以适当引一些paper）

在AutoGL中，我们将图鲁棒性的算法分为三类，放在不同的模块中实现。
鲁棒图特征工程旨在数据预处理阶段生成鲁棒的图特征，增强下游任务的鲁棒性。
鲁棒图神经网络则是通过模型层面的设计，以在训练过程中确保模型的鲁棒性。
鲁棒图神经网络架构搜索旨在搜索出一个鲁棒的图神经网络架构。
下文中将分别介绍这三类图鲁棒性算法。

鲁棒图特征工程
---------------------

我们提供了图结构特征工程的系列方法来提高鲁棒性，请参考`预处理`部分获取更多信息。

鲁棒图神经网络
---------------------

我们提供了一系列的防御方法，旨在增强图神经网络的鲁棒性。

生成并训练 GNNGuard 模型
>>>>>>>>>>>>>>>>>>>

首先，加载预先攻击的图数据：

.. code-block:: python

    perturbed_data = PrePtbDataset(root='/tmp/', name=dataset,attack_method='meta', ptb_rate=0.2)
    modified_adj = perturbed_data.adj

然后，在原图 / 扰动图上训练图神经网络模型：

.. code-block:: python

    flag = False
    print('=== testing GNN on original(clean) graph (AutoGL) ===')
    print("acc_test:",test_autogl(adj, features, device, attention=flag))
    print('=== testing GCN on perturbed graph (AutoGL) ===')
    print("acc_test:",test_autogl(modified_adj, features, device, attention=flag))

训练图神经网络的细节如下：

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

最后，在扰动图上训练防御模型GNNGuard

.. code-block:: python

    flag = True
    print('=== testing GNN on original(clean) graph (AutoGL) + GNNGuard ===')
    print("acc_test:",test_autogl(adj, features, device, attention=flag))
    print('=== testing GCN on perturbed graph (AutoGL) + GNNGuard ===')
    print("acc_test:",test_autogl(modified_adj, features, device, attention=flag))

鲁棒图神经网络架构搜索
---------------------
鲁棒图神经网络架构搜索旨在搜索受到攻击的对抗性鲁棒图神经网络。在AutoGL中，该模块是G-RNA的代码实现。

具体来说，我们通过将邻接矩阵掩码算子添加到搜索空间来为消息传递机制设计一个鲁棒的搜索空间。它受到各种防御性算子的启发，使我们能够搜索到具有防御能力的GNN。

此外，我们定义了一个鲁棒性度量来指导搜索过程，这有助于过滤鲁棒架构。G-RNA使我们能够有效地搜索最优的鲁棒性GNN，并从架构的角度理解GNN鲁棒性。

邻接矩阵掩码算子
受当前防御方法的启发，我们总结了针对图结构的鲁棒性算子。相应地，在我们的搜索空间中设计加入这些代表性防御算子。
这样，我们就可以在面对扰动图时选择最合适的防御策略。

据我们所知，这是首次以增强GNN的鲁棒性为目的而设计搜索空间。

具体来说，我们在搜索空间中包括五个掩码操作:

- 相同算子与前一层保持相同的邻接矩阵
- 低秩近似算子从奇异值分解的前k个分量重建邻接矩阵
- 节点特征相似性算子删除节点特征之间具有小jaccard相似性的边
- 邻居重要性估计算子使用基于量化节点之间相关性的修剪策略更新掩码值
- 可变幂运算符算子从由影响强度参数加权的原始邻接矩阵形成可变幂图


鲁棒性度量
>>>>>>>>>
直觉上，鲁棒的GNN的表现在面对各种扰动时（扰动图）和干净图数据相比不应该变差太多。

我们使用KL距离来衡量干净数据和扰动数据之间的预测差异。

较大的鲁棒性分数表明干净数据和扰动数据的预测之间的距离较小，因此，GNN架构对于潜在的攻击更加鲁棒。


鲁棒图神经网络家沟搜索：G-RNA
>>>>>>>>>>>>>>>>>>>>>>>>>>>
G-RNA能够基于干净的图数据搜索鲁棒的图神经网络，并搜索到的架构在扰动图数据上可以获得**高对抗鲁棒性**。

具体来说，G-RNA过将邻接矩阵掩码算子添加到搜索空间中，允许我们搜索鲁棒GNN。此外，它定义了一个鲁棒性度量来指导搜索过程，这有助于过滤鲁棒架构。通过这种方式，G-RNA能够从架构的角度理解GNN鲁棒性，并有效地搜索最优的对抗性鲁棒GNN。

以下是G-RNA实现的一个例子。

首先，加载相关数据集。

.. code-block:: python

    import os
    os.environ["AUTOGL_BACKEND"] = "pyg"

    # 加载数据集
    from autogl.datasets import build_dataset_from_name
    dataset = build_dataset_from_name('Cora', path='./')


接着，你可以定义自己的GRNA搜索空间和GRNA估计器。

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

或者，直接在节点分类器 ``AutoNodeClassifier``中输入GRNA字符串，使用默认参数。

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

定义好节点分类器之后，可以进行最佳鲁棒架构搜索。

.. code-block:: python

    device = 'cuda'
    solver.fit(dataset)
    solver.get_leaderboard().show()
    orig_acc = solver.evaluate(metric="acc")
    trainer = solver.graph_model_list[0]
    trainer.device = device

最后，针对干净/扰动图进行架构的结果评估。

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
        
    ## 干净图评估
    acc = test_from_data(trainer, dataset)

    ## 扰动图评估
    data = dataset[0].cpu()
    dataset[0] = metattack(data).to(device)
    ptb_acc = test_from_data(trainer, dataset)
