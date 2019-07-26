import logging
import pandas as pd
import numpy as np
import networkx as nx

from collections import defaultdict
from pprint import pformat


class Quanzi(object):

    def __init__(self, threshold=0.3, log_level=logging.INFO):
        super(Quanzi, self).__init__()
        self._input = None
        self._output = None
        self._threshold = threshold

        logging.basicConfig(format='%(asctime)-15s %(levelname)-8s %(message)s', level=log_level)
        self._logger = logging.getLogger('Quanzi')

    def set_input(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        self._input = dataframe

    def read_csv(self, filepath, **kwargs):
        self._input = pd.read_csv(filepath, index_col=0, **kwargs)

    def run(self):
        if self._input.index.dtype == self._input.columns.dtype:
            assert all(i == c for i, c in zip(self._input.index, self._input.columns))
        else:
            assert all(str(i) == c for i, c in zip(self._input.index, self._input.columns))
        self._calc()
        self._logger.debug("Output:\n{}".format(pformat(self._output)))
        return self._output

    def to_csv(self, filepath, **kwargs):
        assert self._output is not None
        df = pd.DataFrame([v for _, v in sorted(self._output.items())])
        df['desc'] = df['desc'].apply(lambda v: ",".join(v))
        df['tag'] = df['tag'].apply(lambda v: ",".join(v))
        df[['name', 'desc', 'tag']].to_csv(filepath, index=False, **kwargs)

    def _calc_guanxi(self, M, n):
        # connections from j with strong connections to colleagues k,
        #   who have strong connections to supervisor i
        # ∑(M[j,k] * M[k, 0])
        S = np.array([np.dot(M[i, 1:], M[1:, 0]) for i in range(1, n)])
        # measures the extent to which actor j is central in the guanxi circle around supervisor i
        G = M[0, 1:].astype(float)
        if S.max() > 0:
            G += (S - S.min()) / (S.max() - S.min())
        self._logger.debug("guanxi is {}".format(G))
        G_std = (G - G.min(axis=0)) / (G.max(axis=0) - G.min(axis=0)) * 6
        self._logger.debug("guanxi std is: {}".format(G_std))
        return G_std

    def _calc_quanzi_leader(self, G):
        def find_cliff(c):
            return np.argmin(abs(G_s[idx_c] - c + G_s[idx_c + 1] - c))
        # ranking G
        idx = G.argsort()[::-1]
        G_s = G[idx]
        # finding "cliffs"
        G_d = G_s[:-1] - G_s[1:]
        idx_c = np.argwhere(G_d > self._threshold)
        # decide class
        # 1. when there are more than two cliffs, divide guanxi to 3 levels:
        #   [2, 4/3], (4/3, 2/3], (2/3, 0]
        # 2. if there is only one cliff, divide guanxi to 2 levels:
        #   [2, 1], (1, 0]
        core_i, peri_i = None, None
        if len(idx_c) >= 2:
            i_4 = idx_c[find_cliff(4)][0]
            i_2 = idx_c[find_cliff(2)][0]
            if G_s[i_4] <= 2:
                peri_i = i_4
            elif G_s[i_2] >= 4:
                core_i = i_2
            else:
                core_i = i_4
                peri_i = i_2
        elif len(idx_c) == 1:
            if G_s[idx_c[0]] >= 3:
                core_i = idx_c[0][0]
            else:
                peri_i = idx_c[0][0]

        results = {
            'core member': (idx[:core_i + 1] + 1).tolist() if core_i else [],
            'peripheral': (idx[core_i + 1:peri_i + 1] + 1) .tolist() if core_i and peri_i else [],
        }
        self._logger.debug("leader quanzi: {}".format(results))
        return results

    def _calc_quanzi_informal_leader(self, M):
        G = nx.from_numpy_matrix(M)
        G.remove_node(0)
        informs = defaultdict(list)
        if G.number_of_edges() == 0:
            self._logger.debug("no quanzi in graph")
            return informs
        for c in nx.k_components(G)[1]:
            if len(c) < 3:
                continue
            H = nx.subgraph(G, c)
            leader = sorted(H.degree, key=lambda x: x[1], reverse=True)[0][0]
            informs[leader] = list(H.neighbors(leader))
        self._logger.debug("informal leader quanzi: {}".format(informs))
        return informs

    def _find_bridge(self, Q, Qs):
        bridges = {}
        m1 = set(Q['core member']).union(Q['peripheral'])
        for k, vs in Qs.items():
            m2 = set(vs + [k])
            bridges[k] = m1.intersection(m2)
        self._logger.debug("bridges: {}".format(bridges))
        return bridges

    def _format(self, Q, Qs, Bridges, n):
        names = dict(zip(range(n), self._input.columns))
        results = defaultdict(lambda: {'desc': [], 'tag': []})
        results[0] = {'desc': ['supervisor'], 'tag': ["0"], 'quanzi-core': [0], 'quanzi': [0]}
        for m in Q['core member']:
            results[m]['desc'].append("core member")
            results[m]['tag'].append("1")
            results[0]['quanzi'].append(m)
            results[0]['quanzi-core'].append(m)
        for m in Q['peripheral']:
            results[m]['desc'].append("peripheral")
            results[m]['tag'].append("2")
            results[0]['quanzi'].append(m)
        for m, ms in Qs.items():
            results[m]['desc'].append("informal leader")
            results[m]['tag'].append("3")
            results[m]['quanzi'] = [m] + ms
            for _m in ms:
                results[_m]['desc'].append("informal leader {}'s core member".format(names[m]))
                results[_m]['tag'].append("4")
        for m, ms in Bridges.items():
            for _m in ms:
                results[_m]['desc'].append('bridge')
                results[_m]['tag'].append("5")
        for m in range(n):
            if m not in results:
                results[m]['desc'].append('outsider')
                results[m]['tag'].append("6")
            results[m]['name'] = names[m]
        self._output = results
