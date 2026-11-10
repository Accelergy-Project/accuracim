"""Command line entry point: `python -m accuracim --config configs/2_scaling-test/r18_dep.yaml`."""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy

from accuracim.config import load_config, project_root
from accuracim.log import log, setup_console, setup_file
from accuracim.pipeline import chunk_params, get_params_list, run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="accuracim", description="CIM noise simulation pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON/Python config")
    parser.add_argument(
        "--mode",
        choices=("both", "estimate", "simulate", "check"),
        help="Run mode override: estimate, simulate, both, or check (is the model converted correctly?)",
    )
    parser.add_argument(
        "--quantization",
        choices=("precise", "noise_source"),
        help="Quantization accounting override: baseline (precise) or additive source",
    )
    parser.add_argument(
        "--group-index", type=int, help="Zero-based chunk of the params list to run"
    )
    parser.add_argument(
        "--num-groups", type=int, default=8, help="Total number of chunks to split into"
    )
    parser.add_argument(
        "--log",
        help="Write the log (and all stdout/stderr) to this file instead of the console "
        "and the default logs/<model>/<timestamp>_<name>.log",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Also show debug details (params, cache folders)"
    )
    return parser.parse_args()


def _select_chunk(config: dict, group_index: int, num_groups: int) -> dict:
    """Narrow `params_grid_list` to one chunk, as singleton grids the pipeline re-expands."""
    params_list = get_params_list(config.get("params_grid_list", []))
    selected = chunk_params(params_list, group_index, num_groups)
    log.info(
        f"Params chunk {group_index + 1}/{num_groups}: "
        f"{len(selected)} of {len(params_list)} params sets"
    )
    config = deepcopy(config)
    config["params_grid_list"] = [
        {key: [value] for key, value in params.items()} for params in selected
    ]
    config["name"] = f"{config.get('name', 'run')}-g{group_index}"
    return config


def main() -> None:
    args = _parse_args()

    # Make `module:func` config paths resolve against your own files next to
    # configs/. `python -m accuracim` puts the cwd here anyway; the `accuracim` console
    # script does not, so do it explicitly and honour CIM_ROOT either way.
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    config = load_config(args.config)

    if args.mode:
        config["run"] = dict(config.get("run", {}), mode=args.mode)
    if args.quantization:
        config["run"] = dict(config.get("run", {}), quantization=args.quantization)

    with ExitStack() as stack:
        if args.log:
            print(f"Logging output to {args.log}", file=sys.stderr)
            log_file = stack.enter_context(open(args.log, "w", buffering=1))
            stack.enter_context(redirect_stdout(log_file))
            stack.enter_context(redirect_stderr(log_file))
            setup_file(args.log, stream=log_file)
        else:
            # The file log (logs/<model>/...) starts once the model is known
            setup_console(verbose=args.verbose)

        if args.group_index is not None:
            config = _select_chunk(config, args.group_index, args.num_groups)

        run_experiment(config)


if __name__ == "__main__":
    main()
