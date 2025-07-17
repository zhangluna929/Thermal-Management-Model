"""Cooling system sub-modules.

This module defines a simple strategy interface for cooling subsystems and a
few concrete implementations. It enables pluggable cooling physics that can be
invoked by the thermal model or a controller (e.g., MPC).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class CoolingSystem(ABC):
    """Abstract base class for cooling strategies."""

    @abstractmethod
    def cooling_power(self, temperatures: np.ndarray) -> np.ndarray:  # W per zone
        """Return cooling power (positive number indicates heat removed)."""


class PassiveCooling(CoolingSystem):
    """No active cooling, only ambient convection already considered."""

    def cooling_power(self, temperatures: np.ndarray) -> np.ndarray:  # noqa: D401
        return np.zeros_like(temperatures)


class LiquidCooling(CoolingSystem):
    """Very simplified liquid cold plate model.

    q = h*A*(T - T_coolant) where h is HTC and A is effective area.
    """

    def __init__(self, htc: float = 50.0, coolant_temp: float = 25.0, area_per_zone: float = 0.005):
        self.htc = htc
        self.coolant_temp = coolant_temp
        self.area = area_per_zone

    def cooling_power(self, temperatures: np.ndarray) -> np.ndarray:
        return self.htc * self.area * (temperatures - self.coolant_temp)


class PCMCooling(CoolingSystem):
    """Phase-change material absorbing latent heat at phase change temperature."""

    def __init__(self, fusion_enthalpy: float = 200e3, pcm_mass_per_zone: float = 0.02, phase_temp: float = 35.0):
        self.fusion_enthalpy = fusion_enthalpy
        self.mass = pcm_mass_per_zone
        self.phase_temp = phase_temp
        self._used_energy = 0.0

    def cooling_power(self, temperatures: np.ndarray) -> np.ndarray:
        power = np.zeros_like(temperatures)
        latent_capacity = self.fusion_enthalpy * self.mass - self._used_energy
        mask = temperatures > self.phase_temp
        # Remove heat limited by remaining latent capacity (simplified per step)
        if latent_capacity > 0 and np.any(mask):
            per_zone = latent_capacity / max(np.count_nonzero(mask), 1)
            power[mask] = per_zone
            self._used_energy += per_zone * np.count_nonzero(mask)
        return power 