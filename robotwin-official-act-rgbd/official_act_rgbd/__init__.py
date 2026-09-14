"""Depth frontends that preserve the official RoboTwin ACT action core."""

from .variants import SUPPORTED_VARIANTS, get_variant

__all__ = ["OfficialACTRGBDPolicy", "SUPPORTED_VARIANTS", "build_policy_and_optimizer", "get_variant"]


def __getattr__(name: str):
    if name in {"OfficialACTRGBDPolicy", "build_policy_and_optimizer"}:
        from .policy import OfficialACTRGBDPolicy, build_policy_and_optimizer

        return {
            "OfficialACTRGBDPolicy": OfficialACTRGBDPolicy,
            "build_policy_and_optimizer": build_policy_and_optimizer,
        }[name]
    raise AttributeError(name)
