"""Model configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class CliffordIPConfig:
    """Architecture and runtime settings for :class:`~cliffordip.wrapper.CliffordIPWrapper`."""

    n_atom_types: int = 100
    n_channels: int = 128
    n_interactions: int = 5
    n_rbf: int = 20
    cutoff: float = 5.0
    n_hidden_output: int = 64
    max_neighbors: int = 50

    use_attention: bool = True
    use_self_interaction: bool = True
    max_body_order: int = 3
    use_multiscale: bool = True
    use_gp_readout: bool = True
    n_heads: int = 4

    use_dens: bool = False
    dens_noise_std: float = 0.01

    use_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # Keys that select the model and its calling convention rather than its architecture.
    _NON_ARCHITECTURE_KEYS = frozenset({"name", "interface"})

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CliffordIPConfig":
        """Build from a config mapping, rejecting keys the model does not accept."""
        known = cls.field_names()
        unknown = set(mapping) - known - cls._NON_ARCHITECTURE_KEYS
        if unknown:
            raise KeyError(
                f"Unknown model config key(s): {', '.join(sorted(unknown))}. "
                f"Accepted: {', '.join(sorted(known))}."
            )
        return cls(**{k: v for k, v in mapping.items() if k in known})

    def replace(self, **changes: Any) -> "CliffordIPConfig":
        return CliffordIPConfig(**{**self.as_dict(), **changes})

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
