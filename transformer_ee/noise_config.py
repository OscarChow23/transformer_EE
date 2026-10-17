"""Validate fractional training-noise settings without importing the training stack.

The CLI and collator share this validator so invalid JSON fails before a training
run starts. Distribution parameters describe epsilon in x -> (1 + epsilon)*x,
not the multiplicative factor itself.
"""

from copy import deepcopy
import math
from numbers import Real


def validate_noise_config(config):
    """Return a validated, default-filled copy of the full config's noise block.

    Missing mode keys retain the historical coherent vector/scalar behavior.
    Keep scalar_mode="auto" symbolic, so changing vector mode later also changes
    its default scalar policy. Never infer that all input features need noise.
    """
    if not isinstance(config.get("noise"), dict):
        raise ValueError("noise must be a configuration object")
    noise = deepcopy(config["noise"])
    for key, default, choices in (
        ("name", "gaussian", ("gaussian", "uniform")),
        ("mode", "coherent", ("coherent", "particle")),
        ("scalar_mode", "auto", ("auto", "none", "coherent")),
    ):
        value = noise.setdefault(key, default)
        if value not in choices:
            raise ValueError(f"noise.{key} must be one of {choices}; got {value!r}")

    # Only the chosen distribution's parameters are active. This permits a CLI
    # distribution override without deleting inactive parameters from JSON.
    defaults = {"mean": 0.0, "std": 0.2} if noise["name"] == "gaussian" else {
        "low": -0.2, "high": 0.2
    }
    for key, default in defaults.items():
        value = noise.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"noise.{key} must be a finite number")
        noise[key] = float(value)
    if noise["name"] == "gaussian" and noise["std"] < 0:
        raise ValueError("noise.std must be nonnegative")
    if noise["name"] == "uniform" and noise["high"] < noise["low"]:
        raise ValueError("noise.high must be greater than or equal to noise.low")

    # Empty lists allow vector-only/scalar-only configurations or deliberate
    # no-op controls. Reject typos and duplicates rather than silently selecting
    # the wrong field or applying an ambiguous repeated advanced index.
    for kind in ("vector", "scalar"):
        names = noise.setdefault(kind, [])
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError(f"noise.{kind} must be a list of feature names")
        if len(set(names)) != len(names):
            raise ValueError(f"noise.{kind} contains duplicate feature names")
        missing = [name for name in names if name not in config.get(kind, [])]
        if missing:
            raise ValueError(f"noise.{kind} contains unknown input features: {missing}")
    return noise
