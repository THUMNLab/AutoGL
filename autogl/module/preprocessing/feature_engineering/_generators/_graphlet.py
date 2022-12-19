import logging
import numpy as np
import torch
from tqdm import tqdm
import autogl
from ._basic import BaseFeatureGenerator
from ..._data_preprocessor_registry import DataPreprocessorUniversalRegistry

_LOGGER = logging.getLogger("FE")


class _Graphlet:
    def __init__(self, data, sample_error=0.1, sample_confidence=0.1):
        self._data = data
        self._init()

        self._sample_error = sample_error
        self._sample_confidence = sample_confidence
        self._dw = int(
            np.ceil(
                0.5 * (self._sample_error ** -2) * np.log(2 / self._sample_confidence)
            )
        )
        _LOGGER.info(
            "sample error {} , confidence {},num {}".format(
                self._sample_error, self._sample_confidence, self._dw
            )
        )

    def _init(self):
        self._edges = list(self._data.edge_index)
        self._edges = [self._edges[0], self._edges[1]]
        self._num_nodes = self._data.x.shape[0]
        self._num_edges = len(self._edges[0])
        self._neighbours = [[] for _ in range(self._num_nodes)]
        for i in range(len(self._edges[0])):
            u, v = self._edges[0][i], self._edges[1][i]
            self._neighbours[u].append(v)

        _LOGGER.info("nodes {} , edges {}".format(self._num_nodes, self._num_edges))

        # sorting
        self._node_degrees = np.array([len(x) for x in self._neighbours])
        self._nodes = np.argsort(self._node_degrees)
        for i in self._nodes:
            self._neighbours[i] = [
                x
                for _, x in sorted(
                    zip(self._node_degrees[self._neighbours[i]], self._neighbours[i]),
                    reverse=True,
                )
            ]
        self._neighbours = [np.array(x) for x in self._neighbours]

    def _get_gdv(self, v, u):
        if self._node_degrees[v] >= self._node_degrees[u]:
            pass
        else:
            u, v = v, u
        Sv, Su, Te = set(), set(), set()
        sigma1, sigma2 = 0, 0
        nb = self._neighbours
        N = self._num_nodes
        M = self._num_edges
        phi = np.zeros(self._num_nodes, dtype=int)
        c1, c2, c3, c4 = 1, 2, 3, 4
        x = np.zeros(16, dtype=int)
        # p1
        for w in nb[v]:
            if w != u:
                Sv.add(w)
                phi[w] = c1
        # p2
        for w in nb[u]:
            if w != v:
                if phi[w] == c1:
                    Te.add(w)
                    phi[w] = c3
                    Sv.remove(w)
                else:
                    Su.add(w)
                    phi[w] = c2
        # p3
        for w in Te:
            for r in nb[w]:
                if phi[r] == c3:
                    x[5] += 1
            phi[w] = c4
            sigma2 = sigma2 + len(nb[w]) - 2
        # p4
        for w in Su:
            for r in nb[w]:
                if phi[r] == c1:
                    x[8] += 1
                if phi[r] == c2:
                    x[7] += 1
                if phi[r] == c4:
                    sigma1 += 1
            phi[w] = 0
            sigma2 = sigma2 + len(nb[w]) - 1
        # p5
        for w in Sv:
            for r in nb[w]:
                if phi[r] == c1:
                    x[7] += 1
                if phi[r] == c4:
                    sigma1 += 1
            phi[w] = 0
            sigma2 = sigma2 + len(nb[w]) - 1

        lsv, lsu, lte, du, dv = len(Sv), len(Su), len(Te), len(nb[u]), len(nb[v])
        # 3-graphlet
        x[1] = lte
        x[2] = du + dv - 2 - 2 * x[1]
        x[3] = N - x[2] - x[1] - 2
        x[4] = N * (N - 1) * (N - 2) / 6 - (x[1] + x[2] + x[3])
        # 4 connected graphlets
        x[6] = x[1] * (x[1] - 1) / 2 - x[5]
        x[10] = lsv * lsu - x[8]
        x[9] = lsv * (lsv - 1) / 2 + lsu * (lsu - 1) / 2 - x[7]
        # 4 disconnected graphlets
        t1 = N - (lte + lsu + lsv + 2)
        x[11] = x[1] * t1
        x[12] = M - (du + dv - 1) - (sigma2 - sigma1 - x[5] - x[8] - x[7])
        x[13] = (lsu + lsv) * t1
        x[14] = t1 * (t1 - 1) / 2 - x[12]
        x[15] = N * (N - 1) * (N - 2) * (N - 3) / 24 - np.sum(x[5:15])

        return x

    def _get_gdv_sample(self, v, u):
        if self._node_degrees[v] >= self._node_degrees[u]:
            pass
        else:
            u, v = v, u
        Sv = set()
        sigma1, sigma2 = 0, 0
        nb = self._neighbours
        N = self._num_nodes
        M = self._num_edges
        phi = np.zeros(self._num_nodes, dtype=int)
        c1, c2, c3, c4 = 1, 2, 3, 4
        x = np.zeros(16)
        dw = self._dw

        # p1
        Sv = set(nb[v][nb[v] != u])
        phi[list(Sv)] = c1
        # p2
        p2w = nb[u][nb[u] != c1]
        p2w1 = p2w[phi[p2w] == c1]
        p2w2 = p2w[phi[p2w] != c1]
        Te = p2w1
        phi[p2w1] = c3
        Sv -= set(list(p2w1))
        Su = p2w2
        phi[p2w2] = c2
        # p3
        for w in Te:
            if dw >= len(nb[w]):
                region = nb[w]
                inc = 1
            else:
                region = np.random.choice(nb[w], dw, replace=False)
                inc = self._node_degrees[w] / dw
            phir = phi[region]
            x[5] += inc * np.sum(phir == c3)
            phi[w] = c4
            sigma2 = sigma2 + len(nb[w]) - 2
        # p4
        for w in Su:
            if dw >= len(nb[w]):
                region = nb[w]
                inc = 1
            else:
                region = np.random.choice(nb[w], dw, replace=False)
                inc = self._node_degrees[w] / dw
            phir = phi[region]
            x[8] += inc * np.sum(phir == c1)
            x[7] += inc * np.sum(phir == c2)
            sigma1 += inc * np.sum(phir == c4)
            phi[w] = 0
            sigma2 = sigma2 + len(nb[w]) - 1
        # p5
        for w in Sv:
            if dw >= len(nb[w]):
                region = nb[w]
                inc = 1
            else:
                region = np.random.choice(nb[w], dw, replace=False)
                inc = self._node_degrees[w] / dw
            phir = phi[region]
            x[7] += inc * np.sum(phir == c1)
            sigma1 += inc * np.sum(phir == c4)
            phi[w] = 0
            sigma2 = sigma2 + len(nb[w]) - 1

        lsv, lsu, lte, du, dv = len(Sv), len(Su), len(Te), len(nb[u]), len(nb[v])
        # 3-graphlet
        x[1] = lte
        x[2] = du + dv - 2 - 2 * x[1]
        x[3] = N - x[2] - x[1] - 2
        x[4] = N * (N - 1) * (N - 2) / 6 - (x[1] + x[2] + x[3])
        # 4 connected graphlets
        x[6] = x[1] * (x[1] - 1) / 2 - x[5]
        x[10] = lsv * lsu - x[8]
        x[9] = lsv * (lsv - 1) / 2 + lsu * (lsu - 1) / 2 - x[7]
        # 4 disconnected graphlets
        t1 = N - (lte + lsu + lsv + 2)
        x[11] = x[1] * t1
        x[12] = M - (du + dv - 1) - (sigma2 - sigma1 - x[5] - x[8] - x[7])
        x[13] = (lsu + lsv) * t1
        x[14] = t1 * (t1 - 1) / 2 - x[12]
        x[15] = N * (N - 1) * (N - 2) * (N - 3) / 24 - np.sum(x[5:15])

        return x

    def get_gdvs(self, sample=True):
        res = np.zeros((self._num_nodes, 15))
        for u in tqdm(range(self._num_nodes)):
            vs = self._neighbours[u]
            if len(vs) != 0:
                gdvs = []
                for v in tqdm(vs, disable=len(vs) < 100):
                    if sample:
                        gdvs.append(self._get_gdv_sample(u, v))
                    else:
                        gdvs.append(self._get_gdv(u, v))
                res[u, :] = np.mean(gdvs, axis=0)[1:]
        return res


@DataPreprocessorUniversalRegistry.register_data_preprocessor("graph" + "let")
class GraphletGenerator(BaseFeatureGenerator):
    r"""generate local graphlet numbers as features. The implementation refers to [#]_ .

    References
    ----------
    .. [#] Ahmed, N. K., Willke, T. L., & Rossi, R. A. (2016).
        Estimation of local subgraph counts. Proceedings - 2016 IEEE International Conference on Big Data, Big Data 2016, 586–595.
        https://doi.org/10.1109/BigData.2016.7840651

    """

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        result: np.ndarray = _Graphlet(data).get_gdvs()
        return torch.from_numpy(result)
