from .quanzi import Quanzi


class UndirectQuanzi(Quanzi):

    def _calc(self):
        Matrix = self._input.to_numpy()
        n = len(Matrix)
        self._logger.debug("matrix: {}".format(Matrix))
        self._calc_guanxi(Matrix, n)
        Q = self._calc_quanzi_leader()
        Qs = self._calc_quanzi_informal_leader(Matrix)
        Bridges = self._find_bridge(Q, Qs)
        self._format(Q, Qs, Bridges, n)
