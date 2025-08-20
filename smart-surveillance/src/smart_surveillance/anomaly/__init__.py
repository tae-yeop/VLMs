"""
Anomaly detection subpackage.

Contains rule-based and learned anomaly detectors that operate over
open-vocabulary detections and simple motion cues.
"""

from .pipeline import run_anomaly

__all__ = ["run_anomaly"]

