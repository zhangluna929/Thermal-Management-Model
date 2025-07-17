"""P2D electrochemical heating computation using PyBAMM.

This is a lightweight wrapper: given a constant current for a short duration,
we return total heat generation distributed evenly across thermal zones.
Future work: run time-resolved coupling and heterogeneous heat mapping.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import pybamm  # type: ignore
except ImportError:  # pragma: no cover
    pybamm = None  # pylint: disable=invalid-name


class PybammNotInstalledError(RuntimeError):
    """Raised when pybamm is required but not installed."""


def compute_heat_generation(
    current: float,
    duration_s: float = 1.0,
    num_zones: int = 3,
    model_name: str = "SPM",
    **model_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Compute heat generation (W) per zone using a PyBAMM cell model.

    Parameters
    ----------
    current : float
        Applied constant current (A). Positive for discharge.
    duration_s : float, optional
        Duration of the step, by default 1 second.
    num_zones : int, optional
        Number of thermal zones to split the heat into, by default 3.
    model_name : str, optional
        One of {"SPM", "DFN"}, default "SPM".
    model_kwargs : dict
        Additional kwargs forwarded to the PyBAMM model constructor.

    Returns
    -------
    np.ndarray
        Heat generation per zone (W).
    """
    if pybamm is None:
        raise PybammNotInstalledError(
            "pybamm is not installed. Install with `pip install pybamm` or disable electrochemical coupling."
        )

    # Select model
    if model_name.upper() == "SPM":
        model = pybamm.lithium_ion.SPM(**model_kwargs)
    elif model_name.upper() == "DFN":
        model = pybamm.lithium_ion.DFN(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Parameter values and simulation
    param = model.default_parameter_values
    # Override current function to constant value in Amps
    C_rate = current / (param["Nominal cell capacity [A.h]"])
    current_func = pybamm.Interpolant(np.array([0, duration_s]), np.array([C_rate, C_rate]))
    param["Current function [A]" if "Current function [A]" in param else "Current function [A]"] = current_func

    sim = pybamm.Simulation(model, parameter_values=param)
    sim.build()
    solution = sim.solve([0, duration_s])

    # Total heating W.m-3 averaged over cell
    if "Total heating [W/m3]" in solution.keys():
        heat_vol = solution["Total heating [W/m3]"].data[-1]
    else:
        heat_vol = solution["Total heating [W.m-3]"].data[-1]

    # Cell volume parameter; fallback approximate cylindrical cell
    cell_volume = param.get("Cell volume [m3]") or (param["Electrode width [m]"] * param["Electrode height [m]"] * param["Electrode thickness [m]"])
    heat_total = heat_vol * cell_volume  # Watts

    # Distribute evenly across zones (placeholder)
    return np.full(num_zones, heat_total / num_zones, dtype=float) 