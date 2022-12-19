.. _trainer_ssl:

AutoGL SSL Trainer
==================

AutoGL project use ``trainer`` to implement the graph self-supervised
methods. Currently, we only support the
`GraphCL <https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html>`__
with semi-supervised downstream tasks:

-  ``GraphCLSemisupervisedTrainer`` using GraphCL algorithm for
   semi-supervised downstream tasks, the main interfaces are shown below

   -  ``train(self, dataset, keep_valid_result=True)``: The function of
      training on the given dataset and keeping valid results

      -  ``dataset``: the graph dataset used to be trained

      -  ``keep_valid_result``: if ``True``, save the validation result
         after training. Only if ``keep_valid_result`` is ``True`` and
         after training, the method ``get_valid_score``,
         ``get_valid_predict_proba`` and ``get_valid_predict`` could
         output meaningful results.

   -  ``predict(self, dataset, mask="test")``: The function of
      predicting on the given dataset

      -  ``dataset``: the graph dataset used to be predicted

      -  ``mask``: ``"train", "val" or "test"``, the dataset mask

   -  ``predict_proba(self, dataset, mask="test", in_log_format=False)``:
      The function of predicting the probability on the given dataset.

      -  ``dataset``: the graph dataset used to be predicted

      -  ``mask``: ``"train", "val" or "test"``, the dataset mask

      -  ``in_log_format``: if ``in_log_format`` is ``True``, the
         probability will be log format

   -  ``evaluate(self, dataset, mask="val", feval=None)``: The function
      of evaluating the model on the given dataset and keeping valid
      result.

      -  ``dataset``: the graph dataset used to be evaluated

      -  ``mask``: ``"train", "val" or "test"``, the dataset mask

      -  ``feval``: the evaluation method used in this function. If
         ``feval`` is ``None``, it will use the ``feval`` given when
         initiate

   -  ``get_valid_score(self, return_major=True)``: The function of
      getting valid scores after training.

      -  ``return_major``: if ``return_major`` is ``True``, then return
         only consists of the major result.

   -  ``get_valid_predict_proba(self)``: Get the prediction probability
      of the valid set after training.

   -  ``get_valid_predict(self)``: Get the valid result after training

Lazy Initialization
-------------------

Similar reason to :ref:model, we also use lazy initialization for all
trainers. Only (part of) the hyper-parameters will be set when
``__init__()`` is called. The ``trainer`` will have its core ``model``
only after ``initialize()`` is explicitly called, which will be done
automatically in ``solver`` and ``duplicate_from_hyper_parameter()``,
after all the hyper-parameters are set properly.

For example, if you want to set ``gcn`` as encoder, a simple ``mlp`` as
a decoder, and use ``mlp`` as a classifier to solve a graph
classification problem, there are three steps you need to do.

-  First, import everything you need

   .. code:: python

      from autogl.module.train.ssl import GraphCLSemisupervisedTrainer
      from autogl.datasets import build_dataset_from_name, utils
      from autogl.datasets.utils.conversion import to_pyg_dataset as convert_dataset

-  Secondly, setup the hyper-parameters of the encoder, decoder and the
   classifier

   .. code:: python

      trainer_hp = {
      	'batch_size': 128,
        'p_lr': 0.0001,             	 # learning rate of pretraining stage
        'p_weight_decay': 0,  				 # weight decay of pretraining stage
        'p_epoch': 100,								 # max epoch of pretraining stage
        'p_early_stopping_round': 100, # early stopping round of pretraining stage
        'f_lr': 0.0001,						  	 # learning rate of fine-tuning stage
        'f_weight_decay': 0,					 # weight decay of fine-tuning stage
        'f_epoch': 100,								 # max epoch of fine-tuning stage
        'f_early_stopping_round': 100, # early stopping round of fine-tuning stage
      }

      encoder_hp = {
        'num_layers': 3,						
        'hidden': [64, 128],					 # hidden dimensions, didn't need to set the dimension of final layer
        'dropout': 0.5,
        'act': 'relu',
        'eps': 'false'
      }

      decoder_hp = {
        'hidden': 64,
        'act': 'relu',
        'dropout': 0.5
      }

      prediction_head_hp = {
        'hidden': 64,
        'act': 'relu',
        'dropout': 0.5
      }

-  Thirdly, use ``duplicate_from_hyper_parameter()``

   .. code:: python

      dataset = build_dataset_from_name('proteins')
      dataset = convert_dataset(dataset)
      utils.graph_random_splits(dataset, train_ratio=0.1, val_ratio=0.1, seed=2022) # split the dataset

      # generate a trainer, but it couldn't be used 
      # before you call `duplicate_from_hyper_parameter`
      trainer = GraphCLSemisupervisedTrainer(
      	model=('gcn', 'sumpoolmlp'),
      	prediction_model_head='sumpoolmlp',
      	views_fn=['random2', 'random2'],
        num_features=dataset[0].x.size(1),
        num_classes=max([data.y.item() for data in dataset]) + 1,
        z_dim=128,	# the embedding dimension
        init=False
      )

      # call duplicate_from_hyper_parameter to set some information about
      # model architecture and learning hyperparameters
      trainer.initialize()
      trainer = trainer.duplicate_from_hyper_parameter(
      	{
          'trainer': trainer_hp,
          'encoder': encoder_hp,
          'decoder': decoder_hp,
          'prediction_head': prediction_head_hp
        }
      )

Train and Predict
-----------------

After initializing a trainer, you can train it on the given datasets.

We are given the training and testing functions for the tasks of graph
classification. You can also create your own tasks following similar
patterns to ours.

We provide some interfaces, and you can easily use them to train or test
on the given datasets.

-  Training: ``train()``

   .. code:: python

      trainer.train(dataset, keep_valid_result=False)

   ``train()`` is the method of training on the given dataset and
   keeping valid results.

   It has two parameters, the first parameter is ``dataset``, which is
   the graph dataset used to be trained. And the second parameter is
   ``keep_valid_result``. It is a bool value, if true, the trainer will
   save the validation result after training if the dataset has a
   validation set.

-  Testing: ``predict()``

   .. code:: python

      trainer.predict(dataset, 'test').detach().cpu().numpy()

   ``predict()`` is the method of predicting the given dataset.

   It has two parameters, the first parameter is ``dataset``, which is
   the graph dataset used to be predicted. And the second parameter is
   ``mask``. It is a string which can be 'train', 'val', or 'test'. And
   returns the prediction results.

-  Evaluation: ``evaluate()``

   .. code:: python

      result = trainer.evaluate(dataset, 'test')    # return a list of metrics, the default metric is accuracy

   ``evaluate()`` is the method of evaluating the model on the given
   dataset and keeping valid results.

   It has three parameters, the first parameter is ``dataset``, which is
   the graph dataset used to be evaluated. And the second parameter is
   ``mask``. It is a string which can be 'train', 'val' or 'test'. And
   the last parameter is ``feval``, which can be a string, tuple of strings,
   or None, which means the used evaluation methods such ``Acc``.

   And you can write your own evaluation metrics and methods. Here is a
   simple example:

   .. code:: python

      from autogl.module.train.evaluation import Evaluation, register_evaluate
      from sklearn.metrics import accuracy_score

      @register_evaluate("my_acc") # use method register_evaluate, and then you can use this class by its register name 'my_acc'
      class MyAcc(Evaluation):
        @staticmethod
        def get_eval_name():
          '''
          define the name, didn't need to same as the registered name
          '''
          return "my_acc"
        
        @staticmethod
        def is_higher_better():
          '''
          return whether this evaluation method is higher better (bool)
          '''
          return True
        
        @staticmethod
        def evaluate(predict, label):
          '''
          return the evaluation result (float)
          '''
          if len(predict.shape) == 2:
          	predict = np.argmax(predict, axis=1)
          else:
          	predict = [1 if p > 0.5 else 0 for p in predict]
          return accuracy_score(label, predict)

Implement SSL Trainer
---------------------

Next, we will show how to implement your own ssl trainer. It is more
difficult to implement the trainer than to use it, it needs to implement
three main functions ``_train_only()``, ``_predict_only()`` and
``duplicate_from_hyper_parameter()``. Now we will implement GraphCL with
unsupervised downstream tasks step by step.

-  initialize your trainer

   First, We need to import some classes and methods, define a basic
   ``__init__()`` method, and register our trainer.

   .. code:: python

      import torch
      from torch.optim.lr_scheduler import StepLR
      from autogl.module.train import register_trainer
      from autogl.module.train.ssl.base import BaseContrastiveTrainer
      from autogl.datasets import utils

      @register_trainer("GraphCLUnsupervisedTrainer")
      class GraphCLUnsupervisedTrainer(BaseContrastiveTrainer):
        def __init__(
          self, 
          model, 
          prediction_model_head, 
          num_features, 
          num_classes, 
          num_graph_features,
          device,
          feval,
          views_fn,
          z_dim,
          num_workers,
          batch_size,
          eval_interval,
          init,
          *args,
          **kwargs,
        ):
          # setup encoder and decoder
          if isinstance(model, Tuple):
            encoder, decoder = model
          elif isinstance(model, BaseAutoModel):
            raise ValueError("The GraphCL trainer must need an encoder and a decoder, so `model` shouldn't be an instance of `BaseAutoModel`")
          else:
            encoder, decoder = model, "sumpoolmlp"
          self.eval_interval = eval_interval
          # init contrastive learning
          super().__init__(
            encoder=encoder,
            decoder=decoder,
            decoder_node=None,
            num_features=num_features,
            num_graph_features=num_graph_features,
            views_fn=views_fn,
            graph_level=True,											# have graph-level features
            node_level=False,											# have node-level features
            device=device,
            feval=feval,				
            z_dim=z_dim,													# the dimension of the embedding output by encoder
            z_node_dim=None,
            *args,
            **kwargs,
          )
          # initialize something specific for your own method
          self.views_fn = views_fn
          self.aug_ratio = aug_ratio
          self._prediction_model_head = None
          self.num_classes = num_classes
          self.prediction_model_head = prediction_model_head
          self.batch_size = batch_size
          self.num_workers = num_workers
          if self.num_workers > 0:
          	mp.set_start_method("fork", force=True)
          # setup the hyperparameter when initializing
          self.hyper_parameters = {
            "batch_size": self.batch_size,
            "p_epoch": self.p_epoch,
            "p_early_stopping_round": self.p_early_stopping_round,
            "p_lr": self.p_lr,
            "p_weight_decay": self.p_weight_decay,
            "f_epoch": self.f_epoch,
            "f_early_stopping_round": self.f_early_stopping_round,
            "f_lr": self.f_lr,
            "f_weight_decay": self.f_weight_decay,
          }
          self.args = args
          self.kwargs = kwargs
          if init:
            self.initialize()

-  ``_train_only(self, dataset)``

   In this method, the trainer trains the model on the given dataset.
   You can define several different methods for different training
   stages.

   -  set the model on the specified device

      .. code:: python

         def _set_model_device(self, dataset):
           self.encoder.encoder.to(self.device)
           self.decoder.decoder.to(self.device)

   -  For training, you can simply call
      ``super(). _train_pretraining_only(dataset, per_epoch)`` to train
      the encoder.

      .. code:: python

         for i, epoch in enumerate(super()._train_pretraining_only(dataset, per_epoch=True)):
           # you can define your own training process if you want
           # for example, we will fine-tune for every eval_interval epoch
           if (i + 1) % self.eval_interval == 0:
             # fine-tuning
             # get dataset
             train_loader = utils.graph_get_split(dataset, "train", batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
             val_loader = utils.graph_get_split(dataset, "val", batch_size=self.batch_size, num_workers=self.num_workers)
             # setup model
             self.encoder.encoder.eval()
             self.prediction_model_head.initialize(self.encoder)
             # just fine-tuning the prediction head
             model = self.prediction_model_head.decoder
             # setup optimizer and scheduler
             optimizer = self.f_optimizer(model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
             scheduler = self._get_scheduler('finetune', optimizer)
             for epoch in range(self.f_epoch):
               model.train()
               for data in train_loader:
                 optimizer.zero_grad()
                 data = data.to(self.device)
                 embeds = self.encoder.encoder(data)
                 out = model(embeds, data)
                 loss = self.f_loss(out, data.y)
                 loss.backward()
                 optimizer.step()
                 if self.f_lr_scheduler_type:
                   scheduler.step()

   -  To implement the full model, we also need to implement the
      ``_predict_only()`` function to evaluate the effect of the model.

      .. code:: python

         def _predict_only(self, loader, return_label=False):
           model = self._compose_model()
           model.eval()
           pred = []
           label = []
           for data in loader:
             data = data.to(self.device)
             out = model(data)
             pred.append(out)
             label.append(data.y)
           ret = torch.cat(pred, 0)
           label = torch.cat(label, 0)
           if return_label:
             return ret, label
           else:
             return ret

   -  ``duplicate_from_hyper_parameter`` is a method that could
      re-generate the trainer. However, if you don't want to use a
      solver to search a good hyper-parameters automatically, you don't
      need to implement it in fact.

      .. code:: python

         def duplicate_from_hyper_parameter(self, hp, encoder="same", decoder="same", prediction_head="same", restricted=True):
         	hp_trainer = hp.get("trainer", {})
             hp_encoder = hp.get("encoder", {})
             hp_decoder = hp.get("decoder", {})
             hp_phead = hp.get("prediction_head", {})
             if not restricted:
               origin_hp = deepcopy(self.hyper_parameters)
               origin_hp.update(hp_trainer)
               hp = origin_hp
             else:
               hp = hp_trainer
             encoder = encoder if encoder != "same" else self.encoder
             decoder = decoder if decoder != "same" else self.decoder
             prediction_head = prediction_head if prediction_head != "same" else self.prediction_model_head
             encoder = encoder.from_hyper_parameter(hp_encoder)
             decoder.output_dimension = tuple(encoder.get_output_dimensions())[-1]
             if isinstance(encoder, BaseEncoderMaintainer) and isinstance(decoder, BaseDecoderMaintainer):
               decoder = decoder.from_hyper_parameter_and_encoder(hp_decoder, encoder)
             if isinstance(encoder, BaseEncoderMaintainer) and isinstance(prediction_head, BaseDecoderMaintainer):
               prediction_head = prediction_head.from_hyper_parameter_and_encoder(hp_phead, encoder)
             ret = self.__class__(
               model=(encoder, decoder),
               prediction_model_head=prediction_head,
               num_features=self.num_features,
               num_classes=self.num_classes,
               num_graph_features=self.num_graph_features,
               device=self.device,
               feval=self.feval,
               loss=self.loss,
               f_loss=self.f_loss,
               views_fn=self.views_fn_opt,
               aug_ratio=self.aug_ratio,
               z_dim=self.last_dim,
               neg_by_crpt=self.neg_by_crpt,
               tau=self.tau,
               model_path=self.model_path,
               num_workers=self.num_workers,
               batch_size=hp["batch_size"],
               eval_interval=self.eval_interval,
               p_optim=self.p_opt_received,
               p_lr=hp["p_lr"],
               p_lr_scheduler_type=self.p_lr_scheduler_type,
               p_epoch=hp["p_epoch"],
               p_early_stopping_round=hp["p_early_stopping_round"],
               p_weight_decay=hp["p_weight_decay"],
               f_optim=self.f_opt_received,
               f_lr=hp["f_lr"],
               f_lr_scheduler_type=self.f_lr_scheduler_type,
               f_epoch=hp["f_epoch"],
               f_early_stopping_round=hp["f_early_stopping_round"],
               f_weight_decay=hp["f_weight_decay"],
               init=True,
               *self.args,
               **self.kwargs
             )

             return ret
