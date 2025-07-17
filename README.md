# 电池热管理模型 | Thermal Management Model

## 项目简介 | Project Overview
本项目由 **lunazhang** 维护，旨在为动力电池与固态电池研发人员提供一套可插拔、可扩展、且足够精细的热管理仿真与控制工具链。它涵盖了从单体到 Pack 级的热-电-控制耦合，既能支撑学术研究的物理深度，也能满足车企工程落地的效率与可维护性。

Maintained by **lunazhang**, this repository offers a pluggable and extensible tool-chain for high-fidelity thermal simulation and control of lithium-ion and solid-state battery packs. The scope ranges from single-cell physics to pack-level thermal-electrochemical-control coupling, balancing academic rigor with the practical requirements of automotive engineering.

## 关键特性 | Key Features
- **多物理耦合** – 内部阻热、反应热 (PyBAMM P2D)、界面接触热阻与辐射对流均被纳入模型。
- **可插拔冷却** – 支持被动散热、液冷板、相变材料 (PCM) 等多种冷却子模块，可在运行时切换。
- **MPC 控制** – 内置基于 CVXPY 的 Model Predictive Control，在保证安全温度的同时最小化能耗。
- **拓扑优化** – 使用 DEAP 实现的遗传算法，可对冷却通道形状或材料参数进行全局搜索。
- **并行参数扫频** – Dask + JAX 向量化与并行计算，快速完成灵敏度分析与不确定性评估。
- **CI/CD & 单元测试** – GitHub Actions 全自动运行 pytest-cov，确保每次提交的物理与数值一致性。
- **交互式可视化** – Plotly 绘制温度-时间曲线，可一键导出 HTML 报告。

- **Multiphysics Coupling** — Accounts for ohmic heat, reaction heat (PyBAMM P2D), interfacial contact resistance and radiative-convective losses.
- **Pluggable Cooling** — Switch between passive convection, liquid cold-plate and PCM modules at run-time.
- **MPC Control** — CVXPY-based Model Predictive Control keeps temperatures safe while minimising energy consumption.
- **Topology Optimisation** — Global search of cooling channel geometry or material parameters via a DEAP genetic algorithm.
- **Parallel Sweeps** — Dask + JAX acceleration enables sensitivity studies and uncertainty quantification in minutes.
- **CI/CD & Tests** — GitHub Actions executes pytest-cov on every push to guarantee physical and numerical consistency.
- **Interactive Visuals** — Plotly dashboards for instant 2D/3D temperature insights and one-click HTML export.

## 快速上手 | Quick Start
```bash
# 依赖安装（建议在虚拟环境中）
pip install -r requirements.txt

# 安装依赖后运行基础仿真（被动散热）
python simulate.py --config configs/battery_default.yaml --current -5 --time_steps 20 --plot

# 启用电化学热源 + 液冷板 + MPC 控制
python simulate.py \
  --config configs/battery_default.yaml \
  --current -10 \
  --cooling liquid \
  --use_electrochem \
  --plot results.html
```
```bash
# Install dependencies (virtual env recommended)
pip install -r requirements.txt

# Passive cooling example
python simulate.py --config configs/battery_default.yaml --current -5 --time_steps 20 --plot

# Electrochemical heat + liquid cooling + MPC
python simulate.py \
  --config configs/battery_default.yaml \
  --current -10 \
  --cooling liquid \
  --use_electrochem \
  --plot results.html
```

## 目录结构 | Directory Layout
```
thermal_model/            # 核心库
 ├── core.py              # 热模型主类
 ├── cooling.py           # 冷却子模块 (Passive/Liquid/PCM)
 ├── electrochem/         # PyBAMM P2D 耦合
 ├── control.py           # MPC 控制器
 ├── ga_optimize.py       # 遗传算法拓扑优化
 ├── fem_coupling.py      # FEniCS 高保真耦合占位
 └── visualization.py     # Plotly 可视化
configs/                  # YAML 电池与工况模板
simulate.py               # 命令行入口
sweep.py                  # Dask 参数扫频脚本
tests/                    # pytest 单元测试
.github/workflows/ci.yml  # CI 配置
```

## 技术路线图 | Roadmap
- **短期** – 完成 FEM 子模型对接、Dash 实时仪表盘与完整英文文档。
- **中期** – 集成实验数据校准、蒙特卡洛不确定性分析、车辆级循环工况库。
- **长期** – HPC 并行 (MPI/FEniCS-x) 与车型定制化模块，形成企业内部标准工具链。

- **Short Term** — Finish FEM coupling, Dash live dashboard, and full English docs.
- **Mid Term** — Experimental calibration, Monte Carlo UQ, and built-in drive-cycle library.
- **Long Term** — HPC parallelisation (MPI/FEniCS-x) and vehicle-specific modules for enterprise deployment.

## 许可协议 | License
本项目基于 **BSD-3-Clause** 开源，可在遵循条款的前提下自由使用、修改与分发。

Released under the **BSD-3-Clause** license, allowing free use, modification and distribution under the stated terms.

## 作者 | Author
- lunazhang 