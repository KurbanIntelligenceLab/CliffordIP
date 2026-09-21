"""Name-to-factory registry shared by models and datasets."""

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps names to factory callables, with reasons recorded for unavailable entries."""

    def __init__(self, kind: str, entry_point_group: str | None = None):
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._factories: dict[str, Callable[..., T]] = {}
        self._unavailable: dict[str, str] = {}
        self._loaded_plugins = False

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator registering ``name`` for the decorated factory."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                existing = self._factories[name]
                raise ValueError(
                    f"{self.kind} '{name}' is already registered by "
                    f"{existing.__module__}.{existing.__qualname__}; "
                    f"{factory.__module__}.{factory.__qualname__} cannot reuse the name."
                )
            self._factories[name] = factory
            self._unavailable.pop(name, None)
            return factory

        return decorator

    def mark_unavailable(self, name: str, reason: str) -> None:
        """Record why ``name`` could not be registered, for use in error messages."""
        if name not in self._factories:
            self._unavailable[name] = reason

    def load_plugins(self) -> None:
        """Register factories advertised by installed packages under the entry-point group."""
        if self._loaded_plugins or self.entry_point_group is None:
            return
        self._loaded_plugins = True
        for ep in entry_points(group=self.entry_point_group):
            try:
                factory = ep.load()
            except Exception as exc:  # one failing plugin must not hide the working ones
                self.mark_unavailable(ep.name, f"plugin failed to load: {exc}")
                continue
            if ep.name not in self._factories:
                self._factories[ep.name] = factory

    def get(self, name: str) -> Callable[..., T]:
        self.load_plugins()
        if name in self._factories:
            return self._factories[name]
        raise KeyError(self._unknown_message(name))

    def names(self) -> list[str]:
        self.load_plugins()
        return sorted(self._factories)

    def unavailable(self) -> dict[str, str]:
        self.load_plugins()
        return dict(sorted(self._unavailable.items()))

    def _unknown_message(self, name: str) -> str:
        parts = [f"Unknown {self.kind} '{name}'."]
        if self._factories:
            parts.append(f"Available: {', '.join(sorted(self._factories))}.")
        else:
            parts.append("Nothing is registered.")
        if self._unavailable:
            detail = "; ".join(f"{k} ({v})" for k, v in sorted(self._unavailable.items()))
            parts.append(f"Unavailable: {detail}.")
        return " ".join(parts)
