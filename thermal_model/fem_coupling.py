"""Placeholder for coupling high-fidelity FEM model (e.g., FEniCS).

Currently provides a dummy function returning zero heat flux. Extend with
actual FEniCS simulation when available.
"""
import numpy as np


def compute_heat_flux(temperatures: np.ndarray) -> np.ndarray:
    """Return additional heat flux per zone (W) from FEM sub-model."""
    return np.zeros_like(temperatures) 