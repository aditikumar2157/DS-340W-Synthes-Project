"""Conformal prediction for aqueous solubility (LogS).

A reproducible evaluation of split conformal prediction on top of a Random
Forest regressor, compared against an uncalibrated ensemble-variance baseline,
under both random and Bemis-Murcko scaffold splits.

Research hypotheses tested by this package (see README):

H1  Conformal calibration produces interval coverage closer to the nominal
    90% target than an uncalibrated baseline.
H2  Point-prediction accuracy is lower under scaffold splitting than under
    random splitting.
H3  Maintaining reliable intervals is harder for structurally unfamiliar
    molecules (low Tanimoto similarity to the training set).
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
