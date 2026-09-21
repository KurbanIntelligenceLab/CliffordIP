"""Bring state dicts saved by earlier versions into the current parameter layout."""

from typing import Any

import torch


def _per_grade_linear(sd: dict) -> dict:
    """Split a single ``weight`` of shape [8, C_out, C_in] into ``w0``-``w3``."""
    out: dict = {}
    for key, val in sd.items():
        if (
            key.endswith(".weight")
            and torch.is_tensor(val)
            and val.dim() == 3
            and val.shape[0] == 8
        ):
            prefix = key[: -len(".weight")]
            out[f"{prefix}.w0"] = val[0]
            out[f"{prefix}.w1"] = val[1:4].mean(0)
            out[f"{prefix}.w2"] = val[4:7].mean(0)
            out[f"{prefix}.w3"] = val[7]
        elif key.endswith(".grade_mask"):
            continue
        else:
            out[key] = val
    return out


def _edge_embed_columns(sd: dict) -> dict:
    """Trim edge embedding weights to the ``n_rbf`` columns the model reads."""
    rbf_key = next((k for k in sd if k.endswith("edge_embed.rbf.offsets")), None)
    if rbf_key is None:
        return sd
    n_rbf = sd[rbf_key].shape[0]
    for net in ("scalar_net", "vector_net"):
        w_key = rbf_key.replace("edge_embed.rbf.offsets", f"edge_embed.{net}.0.weight")
        if w_key in sd and sd[w_key].shape[1] > n_rbf:
            sd[w_key] = sd[w_key][:, :n_rbf].contiguous()
    return sd


def _drop_removed_parameters(sd: dict) -> dict:
    """Remove entries the current model has no parameter for."""
    removed_suffixes = (".b3",)
    return {k: v for k, v in sd.items() if not k.endswith(removed_suffixes)}


def migrate_state_dict(sd: dict) -> dict:
    """Return ``sd`` in the current parameter layout."""
    sd = _per_grade_linear(dict(sd))
    sd = _edge_embed_columns(sd)
    return _drop_removed_parameters(sd)


def migrate_checkpoint_file(path: str, new_path: str | None = None) -> dict[str, Any]:
    """Migrate a checkpoint on disk, optionally writing it to ``new_path``."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict"):
        if key in ckpt:
            ckpt[key] = migrate_state_dict(ckpt[key])
    if new_path is not None:
        torch.save(ckpt, new_path)
    return ckpt
