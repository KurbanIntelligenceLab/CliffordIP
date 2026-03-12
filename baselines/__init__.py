# Model registration — auto-discovers and imports model subpackages
# Each model lives in models/{name}/ with __init__.py and config.yaml

import importlib
import logging
import pathlib

logger = logging.getLogger(__name__)

from importlib.resources import files as _resource_files

_pkg_dir = pathlib.Path(__file__).parent
_configs_dir = pathlib.Path(str(_resource_files("cliffordip").joinpath("configs", "model")))
_registered_models = {}
_failed_models = {}


def discover_and_register_models():
    """
    Auto-discover model directories and register them with error handling.

    A valid model directory must contain:
    - __init__.py (for model registration)
    - config.yaml (in model dir) OR configs/model/{name}_config.yaml

    Returns dict of successfully registered models.
    """
    # Skip common utility directories
    skip_dirs = {"__pycache__", "common", "registry"}

    for subdir in sorted(_pkg_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in skip_dirs:
            continue

        model_name = subdir.name
        init_file = subdir / "__init__.py"
        config_in_model = subdir / "config.yaml"
        config_in_configs = _configs_dir / f"{model_name}_config.yaml"

        # Check for required files
        if not init_file.exists():
            logger.debug(f"Skipping {model_name}: no __init__.py")
            continue

        if not config_in_model.exists() and not config_in_configs.exists():
            logger.warning(f"Skipping {model_name}: no config.yaml")
            continue

        # Try to import the model
        try:
            module = importlib.import_module(f"baselines.{model_name}")
            _registered_models[model_name] = module
            logger.debug(f"Successfully registered model: {model_name}")
        except Exception as e:
            _failed_models[model_name] = str(e)
            logger.warning(f"Failed to import model '{model_name}': {e.__class__.__name__}: {e}")

    # Log summary
    if _registered_models:
        logger.info(
            f"Registered {len(_registered_models)}/{len(_registered_models) + len(_failed_models)} models: "
            f"{', '.join(sorted(_registered_models.keys()))}"
        )

    if _failed_models:
        logger.warning(f"Failed to register {len(_failed_models)} models: {', '.join(sorted(_failed_models.keys()))}")
        logger.warning(
            "Install missing dependencies to enable all models. See README.md or requirements.txt for details."
        )

    return _registered_models


# Auto-discover and register models at import time
discover_and_register_models()


def get_registered_models():
    """Return list of successfully registered model names."""
    return list(_registered_models.keys())


def get_failed_models():
    """Return dict of failed models and their error messages."""
    return _failed_models.copy()
