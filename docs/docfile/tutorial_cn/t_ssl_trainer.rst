AutoGL SSL Trainer
==================

AutoGL 使用 ``trainer`` 来实现图的自监督学习.
目前，我们的项目仅支持半监督学习的下游任务的
`GraphCL <https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html>`__\ 方法:

-  ``GraphCLSemisupervisedTrainer``
   使用GraphCL算法进行半监督的下游任务，主要接口如下

   -  ``train(self, dataset, keep_valid_result=True)``:
      对给定数据集进行训练，并记录验证集上的效果。

      -  ``dataset``: 图数据集

      -  ``keep_valid_result``: 如果为
         ``True``\ ，在训练之后保存验证集上的效果。当且仅当
         ``keep_valid_result`` 为 ``True`` 并且在完成训练之后，方法
         ``get_valid_score``, ``get_valid_predict_proba`` 和
         ``get_valid_predict`` 才会输出合理的结果

   -  ``predict(self, dataset, mask="test")``:
      对给定的数据集进行预测评估

      -  ``dataset``: 图数据集

      -  ``mask``: ``"train", "val" or "test"``

   -  ``predict_proba(self, dataset, mask="test", in_log_format=False)``:
      对给定数据集进行预测评估，并输出分类的概率

      -  ``dataset``: 图数据集

      -  ``mask``: ``"train", "val" or "test"``

      -  ``in_log_format``: 如果 ``in_log_format`` 为
         ``True``\ ，将输出概率的log值

   -  ``evaluate(self, dataset, mask="val", feval=None)``:
      在给定数据集上评估模型并保存验证集上的模型效果。

      -  ``dataset``: 图数据集

      -  ``mask``: ``"train", "val" or "test"``

      -  ``feval``: 在本方法中使用的评估方式。如果 ``feval`` 输入为
         ``None``\ ，则将使用在初始化时传入的\ ``feval``\ 方式

   -  ``get_valid_score(self, return_major=True)``:
      得到验证集上的模型效果

   -  ``get_valid_predict_proba(self)``: 得到验证集上的分类的概率值

   -  ``get_valid_predict(self)``: 得到在验证集上的分类结果

Lazy Initialization
-------------------

与
:ref:model中相似的原因，我们同样使用懒加载来初始化所有的trainer。当\ ``__init__()``\ 方法被调用时，只有部分参数得到了初始化。只有当\ ``initialize()``\ 被调用时，\ ``trainer``\ 才会得到完全的初始化，这一步骤将在\ ``solver``\ 的\ ``duplicate_from_hyper_parameter()``\ 方法中自动进行。

以下是一个简单的例子，如果您想要设置 ``gcn`` 作为encoder，一个简单的
``mlp`` 作为decoder，并使用 ``mlp``
作为分类器来完成图分类任务，您只需要完成三步简单的步骤。

-  首先导入需要的包

   .. code:: python

      from autogl.module.train.ssl import GraphCLSemisupervisedTrainer
      from autogl.datasets import build_dataset_from_name, utils
      from autogl.datasets.utils.conversion import to_pyg_dataset as convert_dataset

-  第二步，设定超参数

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

-  第三部，调用 ``duplicate_from_hyper_parameter()``\ 方法

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
      # model architecture and learning hyper parameters
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

``trainer``\ 的初始化完成之后，您可以在给定的数据集上训练它。

我们给出了图分类任务的训练和测试函数，您也可以按照与我们相似的模式创建您自己的任务。

我们提供了一些接口，您可以使用它们来训练或者测试给定的数据集。

-  训练： ``train()``

   .. code:: python

      trainer.train(dataset, keep_valid_result=False)

   ``train()`` 方法可以用来对给定数据集进行训练。

   它拥有两个参数，第一个参数是
   ``dataset``\ ，代表了需要被训练的数据集。第二个参数是
   ``keep_valid_result``\ ，它是一个布尔值，如果为真并且数据集存在验证集，那么在完成训练后\ ``trainer``\ 将会对验证集的结果进行评估并保存。

-  测试： ``predict()``

   .. code:: python

      trainer.predict(dataset, 'test').detach().cpu().numpy()

   ``predict()`` 方法可以用来对数据集进行测试。

   它拥有两个参数，第一个参数是
   ``dataset``\ ，代表了需要被测试的数据集。第二个参数是 ``mask``.
   它是一个字符串，可选值为'train'，'val'或者'test'，代表需要测试的数据集划分的部分。

-  评估： ``evaluate()``

   .. code:: python

      result = trainer.evaluate(dataset, 'test')    # return a list of metrics, the default metric is accuracy

   ``evaluate()``\ 方法用于评估数据集。

   它拥有三个参数，第一个参数是
   ``dataset``\ ，代表了需要被评估的数据集。第二个参数是 ``mask``.
   它是一个字符串，可选值为'train'，'val'或者'test'，代表需要评估的数据集划分的部分。最后一个参数为
   ``feval``\ ，它可以是一个字符串、一组字符串或者\ ``None``\ ，代表了需要使用的评估方法如\ ``Acc``\ 。

   并且您可以实现自己的评价指标或者方法，以下是一个简单的例子：

   .. code:: python

      from autogl.module.train.evaluation import Evaluation, register_evaluate
      from sklearn.metrics import accuracy_score

      @register_evaluate("my_acc") # use method register_evaluate, and then you can use this class by it's register name 'my_acc'
      class MyAcc(Evaluation):
        @staticmethod
        def get_eval_name():
          '''
          define the name, didn't need to same as the register name
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

接下来我们将展示如何实现你自己的自监督学习训练器。实现训练器比使用它更难，它需要实现三个主要函数\ ``_train_only()``\ ，\ ``_predict_only()``\ 和\ ``duplicate_from_hyper_parameter()``\ 。现在我们将一步步实现GraphCL的无监督下游任务。

-  初始化您的训练器

   首先，我们需要导入一些类和方法，定义一个基本的\ ``__init__()``\ 方法，并注册自定义的\ ``trainer``\ 。

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
          # setup the hyperparameter when initialize
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

   在该方法中，\ ``trainer``\ 在给定的数据集上训练模型。你可以为不同的训练阶段定义几种不同的方法。

   -  指定训练设备

      .. code:: python

         def _set_model_device(self, dataset):
           self.encoder.encoder.to(self.device)
           self.decoder.decoder.to(self.device)

   -  对于训练，您可以简单地调用\ ``super(). _train_pretraining_only(dataset, per_epoch)``
      方法来训练encoder

      .. code:: python

         for i, epoch in enumerate(super()._train_pretraining_only(dataset, per_epoch=True)):
           # you can define your own training process if you want
           # for example, we will fine-tuning for every eval_interval epochs
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

   -  为了实现完整的\ ``trainer``\ ，我们还需要实现\ ``_predict_only()``\ 函数来评估模型的效果。

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

   -  ``duplicate_from_hyper_parameter``\ 是一个可以生成\ ``trainer``\ 的方法。然而，如果你不想用\ ``solver``\ 自动搜索一个好的超参数，事实上你不需要实现它。

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
