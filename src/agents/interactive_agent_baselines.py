"""Interactive Agent Baselines (TAIRA-style, InteRecAgent-style, MACRS-style).

Placeholder stubs for a future milestone. Not implemented in the current scope.
"""
from __future__ import annotations


class _NotImplementedBase:
    implementation_mode = "not_implemented"

    def run_episode(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} is out of scope for the current milestone."
        )


class TAIRAStyleBaseline(_NotImplementedBase):
    pass


class InteRecAgentStyleBaseline(_NotImplementedBase):
    pass


class MACRSStyleBaseline(_NotImplementedBase):
    pass
