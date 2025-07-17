import numpy as np
from typing import List, Union, Optional
from .cooling import CoolingSystem, PassiveCooling


class ThermalManagementModel:
    """Advanced thermal management model for battery packs.

    This version is refactored into a reusable library module.
    Future extensions (electro-thermal coupling, cooling system co-design, etc.)
    can be added as mix-ins or separate sub-modules.
    """

    def __init__(
        self,
        capacity: float,
        internal_resistance: float,
        ambient_temperature: float,
        heat_capacity: float = 2000,
        surface_area: float = 0.01,
        thermal_conductivity: float = 0.5,
        contact_resistance: float = 0.002,
        arrhenius_coeff: float = 0.01,
        convective_heat_transfer_coefficient: float = 10,
        radiative_emissivity: float = 0.9,
        stefan_boltzmann_constant: float = 5.67e-8,
        num_zones: int = 3,
        cooling_system: Optional[CoolingSystem] = None,
    ) -> None:
        """Initialize the thermal model.

        Parameters
        ----------
        capacity: 电池容量 (Ah)
        internal_resistance: 电池内部电阻 (Ω)
        ambient_temperature: 环境温度 (℃)
        heat_capacity: 比热容 (J/kg·℃)
        surface_area: 表面积 (m²)
        thermal_conductivity: 热导率 (W/m·℃)
        convective_heat_transfer_coefficient: 对流换热系数 (W/m²·℃)
        radiative_emissivity: 辐射率 (dimensionless)
        stefan_boltzmann_constant: 斯特藩常数 (W/m²·K⁴)
        num_zones: 分区数量
        """
        self.capacity = capacity
        self.internal_resistance = internal_resistance
        self.ambient_temperature = ambient_temperature
        self.temperature = np.full(num_zones, ambient_temperature, dtype=float)

        # Thermal parameters
        self.heat_capacity = heat_capacity
        self.surface_area = surface_area
        self.thermal_conductivity = thermal_conductivity
        self.contact_resistance = contact_resistance  # K/W per interface
        self.arrhenius_coeff = arrhenius_coeff  # Temperature coefficient for internal resistance
        self.convective_htc = convective_heat_transfer_coefficient
        self.emissivity = radiative_emissivity
        self.sigma = stefan_boltzmann_constant
        self.num_zones = num_zones
        self.cooling_system: CoolingSystem = cooling_system or PassiveCooling()

        # Dynamic arrays per zone
        self.power_loss = np.zeros(num_zones)
        self.cooling_efficiency = np.zeros(num_zones)
        self.specific_heat = np.full(num_zones, 1000.0)  # 可按材料区分
        self.degradation_factor = np.ones(num_zones)
        self.temperature_gradient = np.zeros(num_zones)

    # ---------------------------------------------------------------------
    # Core physics helpers
    # ---------------------------------------------------------------------
    def calculate_internal_heat_generation(self, current: float) -> float:
        """Return I²R heat generation with temperature-dependent internal resistance."""
        avg_temp = float(np.mean(self.temperature))
        dynamic_r = self.internal_resistance * np.exp(self.arrhenius_coeff * (avg_temp - 25.0))
        return (current ** 2) * dynamic_r

    def _zone_heat_losses(self, idx: int) -> float:
        """Compute total heat losses for a single zone (W)."""
        # Conduction to neighbours
        q_cond_left = (
            (self.temperature[idx] - self.temperature[idx - 1]) / self.contact_resistance
            if idx > 0
            else 0.0
        )
        q_cond_right = (
            (self.temperature[idx] - self.temperature[idx + 1]) / self.contact_resistance
            if idx < self.num_zones - 1
            else 0.0
        )

        # Convection with ambient
        q_conv = self.convective_htc * (self.temperature[idx] - self.ambient_temperature)

        # Radiation (converted to °C^4 ≈ K^4 for small ΔT this is acceptable)
        q_rad = self.emissivity * self.sigma * self.surface_area * (
            self.temperature[idx] ** 4 - self.ambient_temperature ** 4
        )

        return q_cond_left + q_cond_right + q_conv + q_rad

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update_temperature(
        self,
        current: float,
        time_step: float,
        external_heat: Union[float, np.ndarray, None] = None,
    ) -> np.ndarray:
        """Update temperature distribution for a single time step.

        Returns
        -------
        np.ndarray
            Updated temperature field per zone.
        """
        # Internal heat (I^2R) plus optional externally supplied heat sources (e.g., from P2D electrochemical model)
        self.power_loss.fill(self.calculate_internal_heat_generation(current))
        if external_heat is not None:
            # Allow scalar or per-zone array
            if np.isscalar(external_heat):
                self.power_loss += external_heat  # broadcast to all zones
            else:
                ext = np.asarray(external_heat, dtype=float)
                if ext.size != self.num_zones:
                    raise ValueError("external_heat length must match num_zones")
                self.power_loss += ext

        for i in range(self.num_zones):
            total_losses = self._zone_heat_losses(i)
            dT = (self.power_loss[i] - total_losses) * time_step / self.specific_heat[i]
            self.temperature[i] += dT * self.degradation_factor[i]
            # Safety clipping
            self.temperature[i] = np.clip(self.temperature[i], self.ambient_temperature, 85.0)
        return self.temperature

    def optimize_cooling(self, max_temperature: float = 45.0) -> List[str]:
        """Apply cooling system and fallback heuristic if needed."""
        # Apply configured cooling system to compute heat removal (W), convert to ΔT
        cooling_power = self.cooling_system.cooling_power(self.temperature)
        # Cooling reduces temperature proportionally to power and step (assume 1s here for simplicity)
        self.temperature -= cooling_power / self.specific_heat

        # Fallback heuristic if still above limit
        strategy: List[str] = []
        for i in range(self.num_zones):
            if self.temperature[i] > max_temperature:
                self.temperature[i] -= 1.0  # additional emergency cooling
                strategy.append("启用冷却")
            else:
                strategy.append("无需冷却")
        return strategy

    def check_battery_status(self) -> np.ndarray:
        status = ["电池过热" if T > 60 else "正常" for T in self.temperature]
        return np.array(status)

    def simulate(
        self,
        current: float,
        time_steps: int = 10,
        time_step: float = 1.0,
        verbose: bool = True,
        external_heat: Union[float, np.ndarray, None] = None,
    ) -> np.ndarray:
        """Run a time-stepping simulation.

        Parameters
        ----------
        current : float
            Applied current (A).
        time_steps : int
            Number of steps to execute.
        time_step : float
            Length of each step (s).
        verbose : bool
            If True, print per-step log.
        external_heat : float | np.ndarray | None
            Constant extra heat source per zone (W). For example, derived from
            an electrochemical model.
        Returns
        -------
        np.ndarray
            Temperature history with shape (time_steps, num_zones).
        """
        history: List[np.ndarray] = []
        for t in range(time_steps):
            temps = self.update_temperature(current, time_step=time_step, external_heat=external_heat)
            cooling = self.optimize_cooling()
            status = self.check_battery_status()
            if verbose:
                print(
                    f"时间: {t}s, 电池区域温度: {temps}, 状态: {status}, 冷却策略: {cooling}"
                )
            history.append(temps.copy())
        return np.stack(history) 