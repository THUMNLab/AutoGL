.. _ensemble_cn:

Ensemble
========

我们现在支持 voting 和 stacking 方法 

Voting
------

Voter本质上构建了base learner预测的加权和。给定一个评估指标，Voter以某种方式确定base learner的权重，使得验证集指标分数最大化。

我们采用Rich Caruana的权重确定方法。该方法首先通过贪婪搜索找到权重相等的(可能是冗余的)base learner集合，然后通过集合中出现的次数指定Voter中的权重。

您可以通过重写 ``_specificy_weights`` 方法来定制自己的权重确定方法。

.. code-block :: python

    # 例子: 对所有base learner 使用同样的权重
    class EqualWeightVoting(Voting):
        def _specify_weights(self, predictions, label, feval):
            return np.ones(self.n_models)/self.n_models
            # 对所有base learner 赋予相同的权重

Stacking
--------

Stacker将Base Learner的预测作为输入来训练元模型，以找到这些base learner的最佳组合。

目前我们支持广义线性模型(GLM)和梯度推进模型(GBM)作为元模型。

创建一个新的ensemble
----------------------

您可以通过继承base ensember,重载``fit``和``ensemble``方法来创建自己的ensember。

.. code-block :: python

    # 例子 : 使用当前可用的最佳模型
    from autogl.module.ensemble.base import BaseEnsembler
    import numpy as np
    class BestModel(BaseEnsembler):
        def fit(self, predictions, label, identifiers, feval):
            if not isinstance(feval, list):
                feval = [feval]
            scores = np.array([feval[0].evaluate(pred, label) for pred in predictions]) * (1 if feval[0].is_higher_better else -1)
            self.scores = dict(zip(identifiers, scores)) # record validation score of base learners
            ensemble_pred = predictions[np.argmax(scores)]
            return [fx.evaluate(ensemble_pred, label) for fx in feval]

        def ensemble(self, predictions, identifiers):
            best_idx = np.argmax([self.scores[model_name] for model_name in identifiers]) # choose the currently best model in the identifiers
            return predictions[best_idx]

