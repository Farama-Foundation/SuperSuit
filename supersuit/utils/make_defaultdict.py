from collections import defaultdict
import warnings


def make_defaultdict(d):
    try:
        dd = defaultdict(type(next(iter(d.values()))))
        for k, v in d.items():
            dd[k] = v
        return dd
    except StopIteration:
        warnings.warn("No agents left in the environment!")
        return {}
