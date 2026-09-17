from __future__ import annotations

import argparse
from pathlib import Path

from performative_optimization.config import load_config
from performative_optimization.experiment import run_benchmark, write_benchmark_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--no-ood", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_benchmark(cfg, include_ood=not args.no_ood)
    write_benchmark_json(result, Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
