import argparse
import yaml
from pathlib import Path

from thermal_model import ThermalManagementModel


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Thermal simulation runner")
    parser.add_argument("--config", required=True, help="YAML file with battery parameters")
    parser.add_argument("--current", type=float, default=-5.0, help="Discharge/charge current (A)")
    parser.add_argument("--time_steps", type=int, default=10, help="Simulation duration in seconds")
    parser.add_argument("--time_step", type=float, default=1.0, help="Single step length (s)")
    parser.add_argument(
        "--use_electrochem",
        action="store_true",
        help="Use PyBAMM electrochemical model to compute additional heat generation",
    )
    parser.add_argument("--plot", nargs="?", const="show", help="Plot temperature history; optionally specify HTML output path")
    parser.add_argument("--cooling", choices=["passive", "liquid", "pcm"], default="passive", help="Choose cooling system type")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from thermal_model.cooling import PassiveCooling, LiquidCooling, PCMCooling

    cooling_map = {
        "passive": PassiveCooling(),
        "liquid": LiquidCooling(),
        "pcm": PCMCooling(),
    }

    model = ThermalManagementModel(**cfg, cooling_system=cooling_map[args.cooling])

    external_heat = None
    if args.use_electrochem:
        try:
            from thermal_model.electrochem import compute_heat_generation

            external_heat = compute_heat_generation(
                current=args.current,
                duration_s=args.time_step,
                num_zones=model.num_zones,
            )
            print(f"Electrochemical heat generation per zone: {external_heat} W")
        except Exception as err:  # pylint: disable=broad-except
            print(f"Failed to compute electrochemical heat generation: {err}")

    history = model.simulate(
        current=args.current,
        time_steps=args.time_steps,
        time_step=args.time_step,
        external_heat=external_heat,
    )

    if args.plot is not None:
        from thermal_model.visualization import plot_temperature_history
        save_path = None if args.plot == "show" else args.plot
        plot_temperature_history(history, save_path)


if __name__ == "__main__":
    main() 