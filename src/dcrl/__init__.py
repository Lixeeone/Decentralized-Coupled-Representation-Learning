"""DCRL: coupled spatial exploration and local subspace learning."""

from .dynamics import DCRL, DynamicsConfig
from .metrics import compute_diagnostics, principal_subspace

__version__ = "0.1.0"
__all__ = ["DCRL", "DynamicsConfig", "compute_diagnostics", "principal_subspace"]
