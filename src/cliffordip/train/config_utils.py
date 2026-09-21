"""Configuration loading with an omegaconf merge chain.

Merge order, later overriding earlier:
    1. configs/base.yaml
    2. configs/dataset/{name}.yaml
    3. configs/model/{name}_config.yaml
    4. --config <path.yaml>
    5. CLI dot-overrides, e.g. training.lr=1e-3
"""

import sys
from importlib.resources import files as _resource_files
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_CONFIGS_DIR: Path = Path(str(_resource_files("cliffordip").joinpath("configs")))


def _find_yaml(subdir: str, name: str) -> Path:
    """Find a YAML file in configs/{subdir}/{name}.yaml."""
    path = _CONFIGS_DIR / subdir / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (_CONFIGS_DIR / subdir).glob("*.yaml"))
        raise FileNotFoundError(f"Config file not found: {path}\nAvailable: {available}")
    return path


def _find_model_config(name: str) -> Path:
    """Find configs/model/{name}_config.yaml."""
    path = _CONFIGS_DIR / "model" / f"{name}_config.yaml"
    if path.exists():
        return path
    available = sorted(
        p.stem.replace("_config", "") for p in (_CONFIGS_DIR / "model").glob("*_config.yaml")
    )
    raise FileNotFoundError(f"Model config not found for '{name}'. Available: {available}")


def load_config(argv=None, defaults=None) -> DictConfig:
    """
    Build a merged config from YAML files and CLI arguments.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments. If None, uses sys.argv[1:].
    defaults : dict or None
        Hard-coded defaults to merge before CLI parsing (useful for
        trainers that pre-set dataset.name).

    Returns the fully merged DictConfig.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Separate --config <file> from dot-overrides
    override_yaml_path = None
    dot_overrides = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            override_yaml_path = argv[i + 1]
            i += 2
        else:
            dot_overrides.append(argv[i])
            i += 1

    # Parse dot-overrides into an OmegaConf dict
    cli_conf = OmegaConf.from_dotlist(dot_overrides) if dot_overrides else OmegaConf.create()

    # Hard-coded defaults from the trainer
    defaults_conf = OmegaConf.create(defaults) if defaults else OmegaConf.create()

    # 1. Base config
    base_path = _CONFIGS_DIR / "base.yaml"
    base_conf = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create()

    # Peek at merged state to resolve dataset and model names
    peek = OmegaConf.merge(base_conf, defaults_conf, cli_conf)

    # 2. Dataset config
    dataset_name = OmegaConf.select(peek, "dataset.name", default=None)
    dataset_conf: Any = OmegaConf.create()
    if dataset_name and isinstance(dataset_name, str):
        dataset_conf = OmegaConf.load(_find_yaml("dataset", dataset_name))

    # 3. Model config (from models/{name}/config.yaml)
    model_name = OmegaConf.select(peek, "model.name", default=None)
    model_conf: Any = OmegaConf.create()
    if model_name and isinstance(model_name, str):
        model_conf = OmegaConf.load(_find_model_config(model_name))

    # 4. Optional override YAML
    override_conf: Any = OmegaConf.create()
    if override_yaml_path:
        override_conf = OmegaConf.load(override_yaml_path)

    # 5. Final merge: base -> defaults -> dataset -> model -> override_yaml -> CLI
    cfg = OmegaConf.merge(
        base_conf, defaults_conf, dataset_conf, model_conf, override_conf, cli_conf
    )

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    assert isinstance(cfg, DictConfig)
    return cfg
