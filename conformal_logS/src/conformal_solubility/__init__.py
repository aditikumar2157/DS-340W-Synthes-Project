"""Conformal prediction for aqueous solubility (LogS).

A reproducible evaluation of conformal prediction on top of a Random Forest
regressor, compared against an uncalibrated ensemble-variance baseline, under
random and Bemis-Murcko scaffold splits, on two datasets (ESOL and AqSolDB).

Research hypotheses tested by this package (see README):

H1  Conformal calibration produces interval coverage closer to the nominal
    90% target than an uncalibrated baseline.
H2  Point-prediction accuracy is lower under scaffold splitting than under
    random splitting.
H3  Maintaining reliable intervals is harder for structurally unfamiliar
    molecules (low Tanimoto similarity to the training set).
H4  Calibrating *within* novelty bins (Mondrian conformal prediction) narrows
    the coverage gap between the least- and most-familiar molecules, which
    H3 shows marginal calibration leaves wide open.
H5  Used to accept or reject screening candidates against a property
    threshold, calibrated intervals make fewer confident mistakes than the
    point prediction a screen would otherwise rely on.

H1-H3 are the hypotheses of the midterm paper. H4 and H5 follow from what
H3 found: marginal coverage is not conditional coverage, and the molecules a
screening campaign cares about are the ones marginal validity hides.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.2.0"
