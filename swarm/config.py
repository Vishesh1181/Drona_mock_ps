"""Loads config.yaml (every parameter lives there) and applies CLI overrides."""
import copy
import yaml


def load_config(path="config.yaml", overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for key, val in (overrides or {}).items():   # "mission.circle_radius": 0.58
        if val is None:
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return cfg
