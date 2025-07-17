"""Parameter sweep CLI using Dask."""
import argparse
from itertools import product

import dask
from dask.distributed import Client, LocalCluster
from ruamel import yaml

from thermal_model import ThermalManagementModel


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_case(params, base_cfg):
    cfg = {**base_cfg, **params}
    model = ThermalManagementModel(**cfg)
    history = model.simulate(current=-5, time_steps=30, verbose=False)
    max_temp = history.max()
    return {*params.items(), ("max_temp", max_temp)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--param", action="append", nargs=3, metavar=("name", "start", "end"), help="Parameter sweep spec")
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    sweep_params = {}
    for name, start, end in args.param:
        sweep_params[name] = (float(start), float(end))

    grid = []
    for key, (lo, hi) in sweep_params.items():
        grid.append([ {key: val} for val in np.linspace(lo, hi, num=5)])

    cases = [dict(item for d in combo for item in d.items()) for combo in product(*grid)]

    cluster = LocalCluster()
    client = Client(cluster)
    futures = [client.submit(run_case, p, base_cfg) for p in cases]
    results = client.gather(futures)
    print(results)


if __name__ == "__main__":
    main() 