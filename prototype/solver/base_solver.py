from pprint import pprint
from prototype.utils.misc import parse_config


class BaseSolver(object):

    def __init__(self, config_file, verbose: bool = False):
        config = parse_config(config_file)
        if verbose:
            pprint(config)

    def setup_envs(self):
        raise NotImplementedError

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
