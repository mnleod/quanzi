import numpy as np
import networkx as nx

from collections import defaultdict
from .quanzi import Quanzi


class UndirectQuanzi(Quanzi):
    def __init__(self, threshold=0.1, **kwargs):
        super(UndirectQuanzi, self).__init__(**kwargs)
        self._threshold = threshold

    def _calc_guanxi(self, M, n):
        # connections from j with strong connections to colleagues k,
        #   who have strong connections to supervisor i
        # ∑(M[j,k] * M[k, 0])
        S = np.array([np.dot(M[i, 1:], M[1:, 0]) for i in range(1, n)])
        # measures the extent to which actor j is central in the guanxi circle around supervisor i
        G = M[0, 1:].astype(float)
        if S.max() > 0:
            G += (S - S.min()) / (S.max() - S.min())
        return G

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
            i_4 = idx_c[find_cliff(4 / 3)][0]
            i_2 = idx_c[find_cliff(2 / 3)][0]
            if G_s[i_4] <= 2 / 3:
                peri_i = i_4
            elif G_s[i_2] >= 4 / 3:
                core_i = i_2
            else:
                core_i = i_4
                peri_i = i_2
        elif len(idx_c) == 1:
            if G_s[idx_c[0]] >= 1:
                core_i = idx_c[0][0]
            else:
                peri_i = idx_c[0][0]

        cores = idx[:core_i + 1] if core_i else np.array([]).astype(int)
        perips = idx[core_i + 1:peri_i + 1] if core_i and peri_i else np.array([]).astype(int)
        return {
            'core member': (cores + 1).tolist(),
            'peripheral': (perips + 1).tolist(),
        }

    def _calc_quanzi_informal_leader(self, M):
        G = nx.from_numpy_matrix(M)
        G.remove_node(0)
        informs = defaultdict(list)
        for c in nx.k_components(G)[1]:
            if len(c) < 3:
                continue
            H = nx.subgraph(G, c)
            leader = sorted(H.degree, key=lambda x: x[1], reverse=True)[0][0]
            informs[leader] = list(H.neighbors(leader))
        return informs

    def _find_bridge(self, Q, Qs):
        bridges = {}
        m1 = set(Q['core member']).union(Q['peripheral'])
        for k, vs in Qs.items():
            m2 = set(vs + [k])
            bridges[k] = m1.intersection(m2)
        return bridges

    def _calc(self):
        Matrix = self._input.to_numpy()
        self._logger.debug("matrix: {}".format(Matrix))
        n = len(Matrix)
        self._logger.debug("matrix size length is {}".format(n))
        Guanxi = self._calc_guanxi(Matrix, n)
        self._logger.debug("guanxi: {}".format(Guanxi))
        Q = self._calc_quanzi_leader(Guanxi)
        self._logger.debug("leader quanzi: {}".format(Q))
        Qs = self._calc_quanzi_informal_leader(Matrix)
        self._logger.debug("informal leader quanzi: {}".format(Qs))
        Bridges = self._find_bridge(Q, Qs)
        self._logger.debug("bridges: {}".format(Bridges))
        self._format(Q, Qs, Bridges, n)

    def _format(self, Q, Qs, Bridges, n):
        names = dict(zip(range(n), self._input.columns))
        results = defaultdict(lambda: {'desc': [], 'tag': [], 'in': [], 'between': []})
        results[0] = {'desc': ['supervisor'], 'tag': ["0"]}
        for m in Q['core member']:
            results[m]['desc'].append("core member")
            results[m]['tag'].append("1")
        for m in Q['peripheral']:
            results[m]['desc'].append("peripheral")
            results[m]['tag'].append("2")
        for m, ms in Qs.items():
            results[m]['desc'].append("informal leader")
            results[m]['tag'].append("3")
            for _m in ms:
                results[_m]['in'].append(m)
                results[_m]['desc'].append("informal leader {}'s core member".format(names[m]))
                results[_m]['tag'].append("4")
        for m, ms in Bridges.items():
            for _m in ms:
                results[_m]['between'].append(m)
                results[_m]['desc'].append('bridge')
                results[_m]['tag'].append("5")
        for m in range(n):
            if m not in results:
                results[m]['desc'].append('outsider')
                results[m]['tag'].append("6")
            results[m]['name'] = names[m]
        self._output = results
