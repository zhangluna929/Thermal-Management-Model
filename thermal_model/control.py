"""Cooling power MPC controller (simplified demo)."""
from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np

from .cooling import CoolingSystem, LiquidCooling


class MPCController:
    """Linear MPC minimizing cooling power while respecting temperature constraints."""

    def __init__(
        self,
        horizon: int = 5,
        max_temp: float = 45.0,
        dt: float = 1.0,
        htc: float = 50.0,
        area: float = 0.005,
    ):
        self.horizon = horizon
        self.max_temp = max_temp
        self.dt = dt
        self.htc = htc
        self.area = area

    def compute_actions(self, temperatures: np.ndarray, ambient: float) -> CoolingSystem:
        zones = temperatures.size
        # Decision variables: coolant_temp deviation below ambient for each step
        delta = cp.Variable((self.horizon, zones))
        cost = cp.sum(cp.square(delta))
        constraints = []
        temp = temperatures.copy()
        for t in range(self.horizon):
            # Cooling power proportional to ΔT between cell and coolant
            power = self.htc * self.area * (temp - (ambient - delta[t]))
            temp = temp - (power * self.dt) / 1000  # simplistic thermal mass
            constraints += [temp <= self.max_temp, delta[t] >= 0, delta[t] <= 20]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        # Use first-step delta to create LiquidCooling with lowered coolant temp
        coolant_temp = ambient - delta.value[0]
        return LiquidCooling(htc=self.htc, coolant_temp=float(np.mean(coolant_temp)), area_per_zone=self.area) 