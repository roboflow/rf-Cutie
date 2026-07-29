import json


def load_subset(path):
    with open(path) as f:
        return set(f.read().splitlines())


def load_empty_masks(path):
    with open(path) as f:
        return json.load(f)
