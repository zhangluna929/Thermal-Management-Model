import numpy as np


class ThermalManagementModel:
    def __init__(self, capacity, internal_resistance, ambient_temperature, heat_capacity=2000, surface_area=0.01,
                 thermal_conductivity=0.5, convective_heat_transfer_coefficient=10, radiative_emissivity=0.9,
                 stefan_boltzmann_constant=5.67e-8, num_zones=3):
        """
        初始化热管理模型，模拟电池的热效应，包括热传导、对流、辐射等。（good luck!）

        参数:
        capacity (float): 电池容量 (Ah)
        internal_resistance (float): 电池内部电阻 (ohm)
        ambient_temperature (float): 环境温度 (°C)
        heat_capacity (float): 电池的比热容 (J/kg·°C)
        surface_area (float): 电池表面积 (m²)
        thermal_conductivity (float): 电池材料的热导率 (W/m·°C)
        convective_heat_transfer_coefficient (float): 对流热传递系数 (W/m²·°C)
        radiative_emissivity (float): 电池表面辐射率
        stefan_boltzmann_constant (float): 斯特藩-玻尔兹曼常数 (W/m²·K⁴)
        num_zones (int): 电池区域数目，用于模拟多温区的热管理
        """
        self.capacity = capacity  # 电池容量 (Ah)
        self.internal_resistance = internal_resistance  # 电池内部电阻 (ohm)
        self.ambient_temperature = ambient_temperature  # 环境温度 (°C)
        self.temperature = np.full(num_zones, ambient_temperature)  # 初始温度 (每个区域)
        self.heat_capacity = heat_capacity  # 电池的比热容 (J/kg·°C)
        self.surface_area = surface_area  # 电池表面积 (m²)
        self.thermal_conductivity = thermal_conductivity  # 热导率 (W/m·°C)
        self.convective_heat_transfer_coefficient = convective_heat_transfer_coefficient  # 对流热传递系数
        self.radiative_emissivity = radiative_emissivity  # 电池表面辐射率
        self.stefan_boltzmann_constant = stefan_boltzmann_constant  # 斯特藩-玻尔兹曼常数
        self.num_zones = num_zones  # 电池区域数目
        self.power_loss = np.zeros(num_zones)  # 电池内部热量产生的功率损失
        self.cooling_efficiency = np.zeros(num_zones)  # 冷却效率（每个区域的冷却能力）
        self.specific_heat = np.full(num_zones, 1000)  # 每个区域的比热容
        self.degradation_factor = np.ones(num_zones)  # 每个区域的退化因子
        self.temperature_gradient = np.zeros(num_zones)  # 每个区域的温度梯度

    def calculate_internal_heat_generation(self, current):
        """
        计算电池各个区域内部的热量生成，基于电流和电池内部电阻。

        参数:
        current (float): 电池的电流 (A)

        返回:
        float: 电池内部的热量产生 (W)
        """
        # 通过内部电阻计算功率损失 (I²R)，每个区域产生的热量是相等的
        heat_generation = (current ** 2) * self.internal_resistance
        return heat_generation

    def update_temperature(self, current, time_step):
        """
        更新电池各个区域的温度，考虑热传导、热对流和热辐射效应。

        参数:
        current (float): 电池的电流 (A)
        time_step (float): 时间步长 (秒)

        返回:
        np.array: 更新后的电池每个区域的温度 (°C)
        """
        self.power_loss.fill(self.calculate_internal_heat_generation(current))  # 每个区域的功率损失

        # 计算热传导、热对流和热辐射效应
        for i in range(self.num_zones):
            # 热传导：邻接区域之间的热量交换
            if i > 0:
                heat_conduction_left = self.thermal_conductivity * (self.temperature[i] - self.temperature[i - 1]) / 2
            else:
                heat_conduction_left = 0
            if i < self.num_zones - 1:
                heat_conduction_right = self.thermal_conductivity * (self.temperature[i] - self.temperature[i + 1]) / 2
            else:
                heat_conduction_right = 0

            # 热对流：与环境的热交换
            heat_convection = self.convective_heat_transfer_coefficient * (
                        self.temperature[i] - self.ambient_temperature)

            # 热辐射：电池表面的辐射热损失
            heat_radiation = self.radiative_emissivity * self.stefan_boltzmann_constant * self.surface_area * (
                    self.temperature[i] ** 4 - self.ambient_temperature ** 4)

            # 综合热损失
            total_heat_loss = heat_conduction_left + heat_conduction_right + heat_convection + heat_radiation
            temperature_change = (self.power_loss[i] - total_heat_loss) * time_step / self.specific_heat[i]

            # 更新电池温度（考虑退化和冷却效应）
            self.temperature[i] += temperature_change * self.degradation_factor[i]  # 退化影响
            self.temperature[i] = np.clip(self.temperature[i], self.ambient_temperature, 85)  # 最大温度限制

        return self.temperature

    def optimize_cooling(self, max_temperature=45):
        """
        优化冷却系统，以确保电池每个区域的温度不超过最大安全温度。

        参数:
        max_temperature (float): 电池的最大安全温度 (°C)

        返回:
        str: 冷却策略 ("启用冷却" 或 "无需冷却")
        """
        cooling_status = []
        for i in range(self.num_zones):
            if self.temperature[i] > max_temperature:
                # 启用冷却，降低温度
                self.temperature[i] -= 2  # 简化：冷却每秒降低2°C
                cooling_status.append("启用冷却")
            else:
                cooling_status.append("无需冷却")
        return cooling_status

    def check_battery_status(self):
        """
        检查电池状态是否异常，如电池温度过高。

        返回:
        np.array: 电池每个区域的状态（"正常"、"电池过热"）
        """
        status = []
        for temp in self.temperature:
            if temp > 60:
                status.append("电池过热")
            else:
                status.append("正常")
        return np.array(status)

    def simulate(self, current, time_steps=10):
        """
        模拟电池在多个时间步骤中的温度变化，并输出每个区域的状态。

        参数:
        current (float): 电池电流 (A)
        time_steps (int): 仿真时长（单位：秒）
        """
        for t in range(time_steps):
            # 更新电池温度
            self.update_temperature(current, time_step=1)

            # 优化冷却系统
            cooling_status = self.optimize_cooling()

            # 检查电池状态
            status = self.check_battery_status()

            # 打印当前状态
            print(f"时间: {t}s, 电池区域温度: {self.temperature}, 状态: {status}, 冷却策略: {cooling_status}")


# 测试热管理模型
if __name__ == "__main__":
    # 创建电池对象，容量50Ah，内部电阻0.1Ω，初始环境温度25°C，模拟3个区域
    battery = ThermalManagementModel(capacity=50, internal_resistance=0.1, ambient_temperature=25, num_zones=3)

    # 模拟电池放电过程
    print("开始仿真：")
    battery.simulate(current=-5, time_steps=10)
