"""Lazy backend selection for TD-MPC2 environments."""

from __future__ import annotations

from typing import Any, Mapping, Optional


SUPPORTED_BACKENDS = ("steve", "nvidia-guided")


def make_env(
    config: Optional[Mapping[str, Any]] = None,
    *,
    backend: Optional[str] = None,
):
    """Build one environment without importing the unselected backend.

    ``backend`` can be supplied explicitly or as ``config["backend"]``.  When
    omitted, the legacy stEVE backend remains the default.
    """

    environment_config = dict(config or {})
    configured_backend = environment_config.pop("backend", None)
    if backend is not None and configured_backend is not None:
        if backend != configured_backend:
            raise ValueError(
                "Conflicting environment backends: explicit backend "
                f"{backend!r}, config backend {configured_backend!r}"
            )
    selected_backend = backend if backend is not None else configured_backend
    if selected_backend is None:
        selected_backend = "steve"
    if not isinstance(selected_backend, str):
        raise TypeError("Environment backend must be a string")

    selected_backend = selected_backend.strip().lower()
    if selected_backend == "steve":
        from .steve_env import make_steve_env

        return make_steve_env(environment_config)
    if selected_backend == "nvidia-guided":
        from .nvidia_guided_env import make_nvidia_guided_env

        return make_nvidia_guided_env(environment_config)
    raise ValueError(
        f"Unsupported environment backend {selected_backend!r}; "
        f"expected one of {SUPPORTED_BACKENDS}"
    )


__all__ = ["SUPPORTED_BACKENDS", "make_env"]
