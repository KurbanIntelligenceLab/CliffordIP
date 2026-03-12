"""
Model registry with decorator-based registration.

Adding a new model:
    1. Create models/{name}/ with __init__.py (registration) and config.yaml
    2. In __init__.py, define build_{name}(cfg) decorated with @register_model("name")
    3. Add the model name to the _model_names list in models/__init__.py

That's it -- zero changes to registry.py, config_utils.py, or any trainer.
"""

from typing import Callable, Dict, List

import torch.nn as nn
from omegaconf import DictConfig

_MODEL_REGISTRY: Dict[str, Callable[[DictConfig], nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model factory function.

    The factory function signature must be:
        def factory(cfg: DictConfig) -> nn.Module

    where cfg is the full merged config (not just the model section).
    """

    def decorator(factory_fn: Callable[[DictConfig], nn.Module]):
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' already registered. "
                f"Existing: {_MODEL_REGISTRY[name].__module__}.{_MODEL_REGISTRY[name].__name__}, "
                f"New: {factory_fn.__module__}.{factory_fn.__name__}"
            )
        _MODEL_REGISTRY[name] = factory_fn
        return factory_fn

    return decorator


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate a model from the registry using the merged config."""
    model_name = cfg.model.name
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: [{available}]. "
            f"Did you forget to add models/{model_name}/?"
        )
    return _MODEL_REGISTRY[model_name](cfg)


def list_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_REGISTRY.keys())
