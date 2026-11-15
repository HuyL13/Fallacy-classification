"""Mechanism-grounded selective contrastive debate, isolated from legacy engines."""

from .pipeline import Pipeline, PipelineConfig
from .schemas import Sample

__all__ = ['Pipeline', 'PipelineConfig', 'Sample']
