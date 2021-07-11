"""
HPO Module for tuning hyper parameters
"""

import time
import numpy as np
from tqdm import trange
from . import register_hpo
from .base import BaseHPOptimizer, TimeTooLimitedError

from .autone_file import utils

from torch_geometric.data import GraphSAINTRandomWalkSampler

from ..feature.graph import SgNetLSD

from torch_geometric.data import InMemoryDataset


class _MyDataset(InMemoryDataset):
    def __init__(self, datalist) -> None:
        super().__init__()
        self.data, self.slices = self.collate(datalist)


@register_hpo("autone")
class AutoNE(BaseHPOptimizer):
    """
    AutoNE HPOptimizer
    The Implementation of "AutoNE: Hyperparameter Optimization for Massive Network Embedding"(KDD 2019).
    See https://github.com/tadpole/AutoNE for more information

    Attributes
    ----------
    max_evals : int
        The max rounds of evaluating HPs
    subgraphs : int
        The number of subgraphs
    sub_evals : int
        The number of evaluation times on each subgraph
    sample_batch_size, sample_walk_length : int
        Using for sampling subgraph, see torch_geometric.data.GraphSAINRandomWalkSampler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get("max_evals", 100)
        self.subgraphs = kwargs.get("subgraphs", 5)
        self.sub_evals = kwargs.get("sub_evals", 5)
        self.sample_batch_size = kwargs.get("sample_batch_size", 150)
        self.sample_walk_length = kwargs.get("sample_walk_length", 2)

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """
        Optimize the HP by the method within give model and HP space

        See .base.BaseHPOptimizer.optimize
        """
        self.feval_name = trainer.get_feval(return_major=True).get_eval_name()
        self.is_higher_better = trainer.get_feval(return_major=True).is_higher_better()
        space = (
            trainer.hyper_parameter_space + trainer.get_model().hyper_parameter_space
        )
        current_space = self._encode_para(space)

        def sample_subgraph(whole_data):
            data = whole_data.data
            loader = GraphSAINTRandomWalkSampler(
                data,
                batch_size=self.sample_batch_size,
                walk_length=self.sample_walk_length,
                num_steps=self.subgraphs,
                save_dir=whole_data.processed_dir,
            )
            results = []
            for data in loader:
                in_dataset = _MyDataset([data])
                results.append(in_dataset)
            return results

        func = SgNetLSD()

        def get_wne(graph):
            graph = func.fit_transform(graph)
            # transform = nx.NxGraph.compose(map(lambda x: x(), nx.NX_EXTRACTORS))
            # print(type(graph))
            # gf = transform.fit_transform(graph).data.gf
            gf = graph.data.gf
            fin = list(gf[0]) + list(map(lambda x: float(x), gf[1:]))
            return fin

        start_time = time.time()

        def fn(dset, para):
            current_trainer = trainer.duplicate_from_hyper_parameter(para)
            current_trainer.train(dset)
            loss, self.is_higher_better = current_trainer.get_valid_score(dset)
            if self.is_higher_better:
                loss = -loss
            return current_trainer, loss

        # code in AutoNE
        sampled_number = self.subgraphs
        k = self.sub_evals
        s = self.max_evals
        X = []
        y = []
        params = utils.Params()
        params.set_space(current_space)
        total_t = 0.0
        info = []
        K = utils.K(len(params.type_))
        gp = utils.GaussianProcessRegressor(K)
        sample_graphs = sample_subgraph(dataset)
        print("Sample Phase:\n")
        for t in trange(sampled_number):
            b_t = time.time()
            i = t
            subgraph = sample_graphs[t]
            wne = get_wne(subgraph)
            for v in range(k):
                kargs = params.random_x()
                para = params.x2dict(kargs)
                externel_para, trial_para = self._decode_para(para)
                _, res = fn(subgraph, externel_para)
                X_reg = params.dict2x(trial_para)
                X.append(np.hstack((X_reg, wne)))
                y.append(res)

        best_res = None
        best_trainer = None
        best_para = None
        wne = get_wne(dataset)
        print("HPO Search Phase:\n")
        for t in trange(s):
            if time.time() - start_time > time_limit:
                self.logger.info("Time out of limit, Epoch: {}".format(str(i)))
                break
            b_t = time.time()
            gp.fit(np.vstack(X), y)
            X_temp, _ = gp.predict(params.get_bound(), params.get_type(), wne)
            X_temp = X_temp[: len(params.type_)]
            para = params.x2dict(X_temp)
            externel_para, trial_para = self._decode_para(para)
            current_trainer, res_temp = fn(dataset, externel_para)
            self._print_info(externel_para, res_temp)
            X_reg = params.dict2x(trial_para)

            X.append(np.hstack((X_reg, wne)))
            y.append(res_temp)
            if not best_res or res_temp < best_res:
                best_res = res_temp
                best_trainer = current_trainer
                best_para = para
            else:
                del current_trainer
            e_t = time.time()
            total_t += e_t - b_t

        if not best_res:
            raise TimeTooLimitedError(
                "Given time is too limited to finish one round in HPO."
            )

        decoded_json, _ = self._decode_para(best_para)
        self.logger.info("Best Parameter:")
        self._print_info(decoded_json, best_res)

        return best_trainer, decoded_json

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        raise NotImplementedError(
            "HP Optimizer must implement the build_hpo_from_args method"
        )
