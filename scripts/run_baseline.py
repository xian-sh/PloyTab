"""Run BLR, ERT, or TabPFN baselines with the repository data layout."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TASK_CONFIG = {
    "ele": {"csv": "Table_Electronic.csv", "smiles": "SMILES"},
    "energy": {"csv": "Table_Energy.csv", "smiles": "SMILES"},
    "barrer": {"csv": "Table_Permeability_Barrer.csv", "smiles": "SMILES"},
    "dft": {"csv": "DFT_properties_simple.csv", "smiles": "SMILES"},
    "md": {"csv": "MD_properties_simple.csv", "smiles": "SMILES"},
    "newmd": {"csv": "polymer_MD.csv", "smiles": "smiles_list"},
    "qc": {"csv": "calculated_polymer_data.csv", "smiles": "psmiles"},
}

KEEP_MODES = ["keep1", "keep2", "keep3", "keep4", "keep25", "keep50", "keep75"]


def build_predictor(args):
    if args.baseline == "blr":
        from polytab.baselines.bayesian_ridge import SimpleBayesianYPredictor

        return SimpleBayesianYPredictor(random_state=args.random_state, test_size=args.test_size)

    if args.baseline == "etr":
        from polytab.baselines.extra_trees import SimpleETRPredictor

        return SimpleETRPredictor(
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            test_size=args.test_size,
        )

    if args.baseline == "tabpfn":
        from polytab.baselines.tabpfn import SimpleTabPFNYPredictor

        return SimpleTabPFNYPredictor(
            device=args.device,
            random_state=args.random_state,
            test_size=args.test_size,
        )

    raise ValueError(f"Unknown baseline: {args.baseline}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run PolyTab baseline models")
    parser.add_argument("--baseline", choices=["blr", "etr", "tabpfn"], required=True)
    parser.add_argument("--task", choices=sorted(TASK_CONFIG), default="md")
    parser.add_argument("--keep-modes", nargs="+", choices=KEEP_MODES, default=["keep1", "keep25", "keep50", "keep75"])
    parser.add_argument("--device", default="cuda", help="Device for TabPFN")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees for ERT")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for ERT")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    task = TASK_CONFIG[args.task]
    csv_path = ROOT / "data" / task["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {csv_path}")

    predictor = build_predictor(args)
    for keep_mode in args.keep_modes:
        predictor.run_pipeline(
            csv_file_path=str(csv_path),
            smiles_column=task["smiles"],
            keep_mode=keep_mode,
            task_key=args.task,
        )


if __name__ == "__main__":
    main()
