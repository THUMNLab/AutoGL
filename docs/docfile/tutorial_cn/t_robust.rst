<<<<<<< HEAD
.. _nas_cn：

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

鲁棒图神经网络
---------------------

鲁棒图神经网络架构搜索
---------------------
=======
==========================
鲁棒模型
==========================

我们提供了一系列的防御方法，旨在增强图神经网络的鲁棒性。

生成并训练 GNNGuard 模型
------------------------------

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
>>>>>>> 200a684ee5167c44f74d7ad704506ecbca7e11d6
