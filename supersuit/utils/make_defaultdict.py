from collections import defaultdict


def make_defaultdict(d):
    try:
        dd = defaultdict(type(next(iter(d.values()))))
        for k, v in d.items():
            dd[k] = v
        return dd
    except StopIteration:
        return {}
