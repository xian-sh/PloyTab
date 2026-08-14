"""Command-line entry point for PolyTab experiments."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TASKS = ["barrer", "dft", "ele", "energy", "md", "newmd", "qc"]
DEFAULT_KEEP_MODES = ["keep1", "keep25", "keep50", "keep75"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PolyTab experiments")
    parser.add_argument("--task", choices=TASKS, default="ele")
    parser.add_argument(
        "--keep-modes",
        nargs="+",
        choices=DEFAULT_KEEP_MODES,
        default=DEFAULT_KEEP_MODES,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from polytab.pipeline import run_experiment

    run_experiment(task_key=args.task, keep_modes=args.keep_modes)


if __name__ == "__main__":
    main()
