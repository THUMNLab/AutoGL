.. _model:

AutoGL Model
============

In AutoGL, we use ``model`` and ``automodel`` to define the logic of graph nerual networks and make it compatible with hyper parameter optimization. Currently we support the following models for given tasks.

+----------------------+----------------------------+
| Tasks                | Models                     |
+======================+============================+
| Node Classification  | ``gcn``, ``gat``, ``sage`` |
+----------------------+----------------------------+
| Graph Classification | ``gin``, ``topk``          |
+----------------------+----------------------------+
| Link Prediction      | ``gcn``, ``gat``, ``sage`` |
+----------------------+----------------------------+

Lazy Initialization
-------------------

In current AutoGL pipeline, some important hyper-parameters related with model cannot be set outside before the pipeline (e.g. input dimensions, which can only be caluclated during running after feature engineered). Therefore, in ``automodel``, we use lazy initialization to initialize the core ``model``. When the ``automodel`` initialization method ``__init__()`` is called with argument ``init`` be ``False``, only (part of) the hyper-parameters will be set. The ``automodel`` will have its core ``model`` only after ``initialize()`` is explicitly called, which will be done automatically in ``solver`` and ``from_hyper_parameter()``, after all the hyper-parameters are set properly.

Define your own model and automodel
-----------------------------------

We highly recommend you to define both ``model`` and ``automodel``, although you only need your ``automodel`` to communicate with ``solver`` and ``trainer``. The ``model`` will be responsible for the parameters initialization and forward logic declaration, while the ``automodel`` will be responsible for the hyper-parameter definiton and organization.

General customization
^^^^^^^^^^^^^^^^^^^^^

Let's say you want to implement a simple MLP for node classification and want to let AutoGL find the best hyper-parameters for you. You can first define the logics assuming all the hyper-parameters are given.

.. code-block:: python

    import torch

    # define mlp model, need to inherit from torch.nn.Module
    class MyMLP(torch.nn.Module):
        # assume you already get all the hyper-parameters
        def __init__(self, in_channels, num_classes, layer_num, dim):
            super().__init__()
            if layer_num == 1:
                ops = [torch.nn.Linear(in_channels, num_classes)]
            else:
                ops = [torch.nn.Linear(in_channels, dim)]
                for i in range(layer_num - 2):
                    ops.append(torch.nn.Linear(dim, dim))
                ops.append(torch.nn.Linear(dim, num_classes))
        
            self.core = torch.nn.Sequential(*ops)
        
        # this method is required
        def forward(self, data):
            # data: torch_geometric.data.Data
            assert hasattr(data, 'x'), 'MLP only support graph data with features'
            x = data.x
            return torch.nn.functional.log_softmax(self.core(x))


After you define the logic of ``model``, you can now define your ``automodel`` to manage the hyper-parameters.

.. code-block:: python

    from autogl.module.model import BaseModel
    
    # define your automodel, need to inherit from BaseModel
    class MyAutoMLP(BaseModel):
        def __init__(self):
            # (required) make sure you call __init__ of super with init argument properly set.
            # if you do not want to initialize inside __init__, please pass False.
            super().__init__(init=False)

            # (required) define the search space
            self.space = [
                {'parameterName': 'layer_num', 'type': 'INTEGER', 'minValue': 1, 'maxValue': 5, 'scalingType': 'LINEAR'},
                {'parameterName': 'dim', 'type': 'INTEGER', 'minValue': 64, 'maxValue': 128, 'scalingType': 'LINEAR'}
            ]

            # set default hyper-parameters
            self.layer_num = 2
            self.dim = 72

            # for the hyper-parameters that are related with dataset, you can just set them to None
            self.num_classes = None
            self.num_features = None

            # (required) since we don't know the num_classes and num_features until we see the dataset,
            # we cannot initialize the models when instantiated. the initialized will be set to False.
            self.initialized = False

            # (required) set the device of current auto model
            self.device = torch.device('cuda')

        # (required) get current hyper-parameters of this automodel
        # need to return a dictionary whose keys are the same with self.space
        def get_hyper_parameter(self):
            return {
                'layer_num': self.layer_num,
                'dim': self.dim
            }
        
        # (required) override to interact with num_classes
        def get_num_classes(self):
            return self.num_classes
        
        # (required) override to interact with num_classes
        def set_num_classes(self, n_classes):
            self.num_classes = n_classes
        
        # (required) override to interact with num_features
        def get_num_features(self):
            return self.num_features
        
        # (required) override to interact with num_features
        def set_num_features(self, n_features):
            self.num_features = n_features

        # (required) instantiate the core MLP model using corresponding hyper-parameters
        def initialize(self):
            # (required) you need to make sure the core model is named as `self.model`
            self.model = MyMLP(
                in_channels = self.num_features,
                num_classes = self.num_classes,
                layer_num = self.layer_num,
                dim = self.dim
            ).to(self.device)

            self.initialized = True
        
        # (required) override to create a copy of model using provided hyper-parameters
        def from_hyper_parameter(self, hp):
            # hp is a dictionary that contains keys and values corrsponding to your self.space
            # in this case, it will be in form {'layer_num': XX, 'dim': XX}
            
            # create a new instance
            ret = self.__class__()

            # set the hyper-parameters related to dataset and device
            ret.num_classes = self.num_classes
            ret.num_features = self.num_features
            ret.device = self.device

            # set the hyper-parameters according to hp
            ret.layer_num = hp['layer_num']
            ret.dim = hp['dim']

            # initialize it before returning
            ret.initialize()

            return ret
        

Then, you can use this node classification model as part of AutoNodeClassifier ``solver``.

.. code-block :: python

    from autogl.solver import AutoNodeClassifier

    solver = AutoNodeClassifier(graph_models=(MyAutoMLP(),))


The model for graph classification is generally the same, except that you can now also receive the ``num_graph_features`` (the dimension of the graph-level feature) through overriding ``set_num_graph_features(self, n_graph_features)`` of ``BaseModel``. Also, please remember to return graph-level logits instead of node-level one in ``forward()`` of ``model``.

Model for link prediction
^^^^^^^^^^^^^^^^^^^^^^^^^

For link prediction, the definition of model is a bit different with the common forward definition. You need to implement the ``lp_encode(self, data)`` and ``lp_decode(self, x, pos_edge_index, neg_edge_index)`` to interact with ``LinkPredictionTrainer`` and ``AutoLinkPredictor``. Taking the class ``MyMLP`` defined above for example, if you want to perform link prediction:

.. code-block:: python

    class MyMLPForLP(torch.nn.Module):
        # num_classes is removed since it is invalid for link prediction
        def __init__(self, in_channels, layer_num, dim):
            super().__init__()
            ops = [torch.nn.Linear(in_channels, dim)]
            for i in range(layer_num - 1):
                ops.append(torch.nn.Linear(dim, dim))
        
            self.core = torch.nn.Sequential(*ops)

        # (required) for interaction with link prediction trainer and solver
        def lp_encode(self, data):
            return self.core(data.x)

        # (required) for interaction with link prediction trainer and solver
        def lp_decode(self, x, pos_edge_index, neg_edge_index):
            # first, get all the edge_index need calculated
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            # then, use dot-products to calculate logits, you can use whatever decode method you want
            logits = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=-1)
            return logits

    class MyAutoMLPForLP(MyAutoMLP):
        def initialize(self):
            # init MyMLPForLP instead of MyMLP
            self.model = MyMLPForLP(
                in_channels = self.num_features,
                layer_num = self.layer_num,
                dim = self.dim
            ).to(self.device)

            self.initialized = True


Model with sampling support
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Towards efficient representation learning on large-scale graph, AutoGL currently support node classification using sampling techniques including node-wise sampling, layer-wise sampling, and graph-wise sampling. See more about sampling in :ref:`trainer`.

In order to conduct node classification using sampling technique with your custom model, further adaptation and modification are generally required.
According to the Message Passing mechanism of Graph Neural Network (GNN), numerous nodes in the multi-hop neighborhood of evaluation set or test set are potentially involved to evaluate the GNN model on large-scale graph dataset.
As the representations for those numerous nodes are likely to occupy large amount of computational resource, the common forwarding process is generally infeasible for model evaluation on large-scale graph.
An iterative representation learning mechanism is a practical and feasible way to evaluate **Sequential Model**,
which only consists of multiple sequential layers, with each layer taking a ``Data`` aggregate as input. The input ``Data`` has the same functionality with ``torch_geometric.data.Data``, which conventionally provides properties ``x``, ``edge_index``, and optional ``edge_weight``.
If your custom model is composed of concatenated layers, you would better make your model inherit ``ClassificationSupportedSequentialModel`` to utilize the layer-wise representation learning mechanism to efficiently conduct representation learning for your custom sequential model.

.. code-block:: python

    import autogl
    from autogl.module.model.base import ClassificationSupportedSequentialModel

    # override Linear so that it can take graph data as input
    class Linear(torch.nn.Linear):
        def forward(self, data):
            return super().forward(data.x)

    class MyMLPSampling(ClassificationSupportedSequentialModel):
        def __init__(self, in_channels, num_classes, layer_num, dim):
            super().__init__()
            if layer_num == 1:
                ops = [Linear(in_channels, num_classes)]
            else:
                ops = [Linear(in_channels, dim)]
                for i in range(layer_num - 2):
                    ops.append(Linear(dim, dim))
                ops.append(Linear(dim, num_classes))

            self.core = torch.nn.ModuleList(ops)

        # (required) override sequential_encoding_layers property to interact with sampling
        @property
        def sequential_encoding_layers(self) -> torch.nn.ModuleList:
            return self.core
        
        # (required) define the encode logic of classification for sampling
        def cls_encode(self, data):
            # if you use sampling, the data will be passed in two possible ways,
            # you can judge it use following rules
            if hasattr(data, 'edge_indexes'):
                # the edge_indexes are a list of edge_index, one for each layer
                edge_indexes = data.edge_indexes
                edge_weights = [None] * len(self.core) if getattr(data, 'edge_weights', None) is None else data.edge_weights
            else:
                # the edge_index and edge_weight will stay the same as default
                edge_indexes = [data.edge_index] * len(self.core)
                edge_weights = [getattr(data, 'edge_weight', None)] * len(self.core)

            x = data.x
            for i in range(len(self.core)):
                data = autogl.data.Data(x=x, edge_index=edge_indexes[i])
                data.edge_weight = edge_weights[i]
                x = self.sequential_encoding_layers[i](data)
            return x

        # (required) define the decode logic of classification for sampling
        def cls_decode(self, x):
            return torch.nn.functional.log_softmax(x)

