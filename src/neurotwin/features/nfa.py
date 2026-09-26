"""NFA-named entry point for the independent MEA spike-list feature engine."""

from .mea import FEATURES, extract_features

__all__ = ["FEATURES", "extract_features"]
