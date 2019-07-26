import logging
import pandas as pd

from pprint import pformat


class Quanzi(object):

    def __init__(self, log_level=logging.INFO):
        super(Quanzi, self).__init__()
        self._input = None
        self._output = None
        self._threshold = None

        logging.basicConfig(format='%(asctime)-15s %(levelname)-8s %(message)s', level=log_level)
        self._logger = logging.getLogger('Quanzi')

    def read_csv(self, filepath, **kwargs):
        self._input = pd.read_csv(filepath, index_col=0, **kwargs)
        assert all(i == c for i, c in zip(self._input.index, self._input.columns))

    def run(self):
        assert isinstance(self._input, pd.DataFrame), "Input must be a pandas DataFrame"
        self._calc()
        self._logger.debug("Output:\n{}".format(pformat(self._output)))
        return self._output

    def to_csv(self, filepath, **kwargs):
        assert self._output is not None, "Output "
        df = pd.DataFrame([v for _, v in sorted(self._output.items())])
        df['desc'] = df['desc'].apply(lambda v: ",".join(v))
        df['tag'] = df['tag'].apply(lambda v: ",".join(v))
        df[['name', 'desc', 'tag']].to_csv(filepath, index=False, **kwargs)
