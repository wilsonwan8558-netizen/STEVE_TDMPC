"""Lazy environment adapters for the available simulation backends."""

from .factory import SUPPORTED_BACKENDS, make_env


def __getattr__(name: str):
    """Retain legacy imports without eagerly loading SOFA or NVIDIA."""

    if name in {"StEVEEnv", "make_steve_env"}:
        from .steve_env import StEVEEnv, make_steve_env

        return {"StEVEEnv": StEVEEnv, "make_steve_env": make_steve_env}[name]
    if name in {"NvidiaGuidedEnv", "make_nvidia_guided_env"}:
        from .nvidia_guided_env import NvidiaGuidedEnv, make_nvidia_guided_env

        return {
            "NvidiaGuidedEnv": NvidiaGuidedEnv,
            "make_nvidia_guided_env": make_nvidia_guided_env,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SUPPORTED_BACKENDS",
    "make_env",
    "StEVEEnv",
    "make_steve_env",
    "NvidiaGuidedEnv",
    "make_nvidia_guided_env",
]
