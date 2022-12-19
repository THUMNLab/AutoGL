.. _robust:

Robust Model
============

We provides a series of defense methods that aim to enhance the robustness of GNNs.

Requirements
------------

During the evaluation, the adversarial attacks on graph are performed by DeepRobust from MSU, please install it by

.. code-block:: bash

git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install

To better plugin GNNGuard to AutoGL, we slightly revised some functions in geometric. Please use the file AutoGL/autogl/module/model/pyg/robust/nn/conv/gcn_conv.py we provided to replace the corresponding file in the installed geometric folder (for example, the folder path could be /home/username/.local/lib/python3.5/site-packages/torch_geometric/nn/conv/).

Note: Don't forget to backup all the original files when you replacing anything, in case you need them at other places!

Building GNNGuard Module
------------------------

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





