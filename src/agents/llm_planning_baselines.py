"""LLM Planning Baselines — placeholder stubs for a future milestone.

These classes are intentionally not implemented in the current InterRec milestone.
They are listed here to define the interface and clarify scope.
"""
from __future__ import annotations


class _NotImplementedBase:
    implementation_mode = "not_implemented"

    def recommend(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} is out of scope for the current milestone."
        )


class ZeroShotLLMRecommender(_NotImplementedBase):
    pass


class OneShotLLMRecommender(_NotImplementedBase):
    pass


class ChainOfThoughtLLMRecommender(_NotImplementedBase):
    pass


class PlanSolveLLMRecommender(_NotImplementedBase):
    pass


class ReActLLMRecommender(_NotImplementedBase):
    pass


class ReflexionLLMRecommender(_NotImplementedBase):
    pass
