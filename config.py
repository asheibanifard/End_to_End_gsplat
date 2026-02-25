"""
neuro3dgs/config.py

YAML config loader with:
  - Nested dot-notation CLI overrides  (e.g. loss.photometric.weight=2.0)
  - Type-preserving string casting
  - Config schema validation
  - Convenience attribute-access wrapper (cfg.render.H)

Usage in train.py:
    from config import load_config
    cfg = load_config("configs/default.yaml", overrides=sys.argv[2:])
    print(cfg.render.H)
    print(cfg["optim"]["lr"]["means"])
"""

from __future__ import annotations
import os
import copy
from pathlib import Path
from typing import Any


# ═════════════════════════════════════════════════════════════════════════════
#  Attribute-access dict wrapper
# ═════════════════════════════════════════════════════════════════════════════

class ConfigNode:
    """
    Wraps a nested dict so values are accessible via both
    attribute access (cfg.render.H) and item access (cfg["render"]["H"]).
    Mutations propagate to the underlying dict.
    """

    def __init__(self, d: dict):
        object.__setattr__(self, "_data", d)

    def __getattr__(self, key: str):
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"Config has no key '{key}'")
        v = data[key]
        return ConfigNode(v) if isinstance(v, dict) else v

    def __setattr__(self, key: str, value: Any):
        object.__getattribute__(self, "_data")[key] = value

    def __getitem__(self, key):
        v = object.__getattribute__(self, "_data")[key]
        return ConfigNode(v) if isinstance(v, dict) else v

    def __setitem__(self, key, value):
        object.__getattribute__(self, "_data")[key] = value

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_data")

    def __repr__(self):
        return f"ConfigNode({object.__getattribute__(self, '_data')})"

    def to_dict(self) -> dict:
        return copy.deepcopy(object.__getattribute__(self, "_data"))


# ═════════════════════════════════════════════════════════════════════════════
#  YAML loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("Install PyYAML: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Dot-notation CLI override parser
# ═════════════════════════════════════════════════════════════════════════════

def _cast(value: str) -> Any:
    """Attempt to cast a string CLI value to int, float, bool, None, or list."""
    if value.lower() == "null" or value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    # List: "[1,2,3]" or "1,2,3"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [_cast(x.strip()) for x in inner.split(",") if x.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value  # keep as string


def _apply_override(cfg: dict, dotpath: str, value: str) -> None:
    """
    Apply a single dot-notation override.
    E.g. "loss.photometric.weight=2.0"  →  cfg["loss"]["photometric"]["weight"] = 2.0
    """
    keys = dotpath.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = _cast(value)


def parse_overrides(args: list[str]) -> dict:
    """
    Parse a list of 'key=value' strings (dot-notation keys) into a nested dict.
    Non-'key=value' strings are silently skipped.
    """
    overrides: dict = {}
    for arg in args:
        if "=" not in arg:
            continue
        dotpath, _, value = arg.partition("=")
        _apply_override(overrides, dotpath.strip(), value.strip())
    return overrides


# ═════════════════════════════════════════════════════════════════════════════
#  Schema validation (lightweight)
# ═════════════════════════════════════════════════════════════════════════════

# Required top-level sections and a sampling of required leaf keys
_REQUIRED = {
    "data":   ["tif", "swc"],
    "init":   ["feature_dim", "max_seeds"],
    "render": ["H", "W", "fov_deg"],
    "optim":  ["iterations"],
    "loss":   [],
    "logging":["out_dir"],
}


def _validate(cfg: dict):
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"Config missing required section: [{section}]")
        for k in keys:
            if k not in cfg[section]:
                raise ValueError(f"Config missing required key: {section}.{k}")

    # Type checks on a few critical numeric fields
    assert isinstance(cfg["render"]["H"], int) and cfg["render"]["H"] > 0
    assert isinstance(cfg["render"]["W"], int) and cfg["render"]["W"] > 0
    assert isinstance(cfg["optim"]["iterations"], int) and cfg["optim"]["iterations"] > 0
    assert 0.0 <= cfg["loss"]["photometric"]["lambda_dssim"] <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

def load_config(
    yaml_path: str | Path,
    overrides: list[str] | None = None,
    validate: bool = True,
) -> ConfigNode:
    """
    Load a YAML config file and apply optional dot-notation CLI overrides.

    Parameters
    ----------
    yaml_path  : path to .yaml config file
    overrides  : list of "section.key=value" strings (from sys.argv)
    validate   : run lightweight schema check

    Returns
    -------
    ConfigNode  — nested attribute-accessible config object

    Example
    -------
    >>> cfg = load_config("configs/default.yaml",
    ...                   overrides=["render.H=1024", "optim.lr.means=1e-4"])
    >>> print(cfg.render.H)          # 1024
    >>> print(cfg.optim.lr.means)    # 0.0001
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = _load_yaml(path)

    if overrides:
        override_dict = parse_overrides(overrides)
        raw = _deep_merge(raw, override_dict)

    if validate:
        _validate(raw)

    return ConfigNode(raw)


def save_config(cfg: ConfigNode, path: str | Path):
    """Serialise a (potentially mutated) ConfigNode back to YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("Install PyYAML: pip install pyyaml")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)


def print_config(cfg: ConfigNode, indent: int = 0):
    """Pretty-print the config to stdout."""
    data = object.__getattribute__(cfg, "_data")
    for k, v in data.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(ConfigNode(v), indent + 2)
        else:
            print(" " * indent + f"{k}: {v}")