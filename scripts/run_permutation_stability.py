import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from polytab.pipeline import create_predictor, resolve_task_config


def summarize(results):
    rows = []
    for name, metrics in results.items():
        vals = {"property": name}
        vals.update(metrics)
        rows.append(vals)
    df = pd.DataFrame(rows)
    r2_col = "R2" if "R2" in df else "R²"
    return {"mean_R2": float(df[r2_col].mean()), "mean_MAE": float(df["MAE"].mean()), "n_properties": int(len(df))}


def make_variant(csv_path, smiles_col, mode, seed, out_dir):
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    if mode == "row":
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    elif mode == "column":
        prop_cols = [c for c in df.columns if c != smiles_col]
        df = df[[smiles_col] + list(rng.permutation(prop_cols))]
    else:
        raise ValueError(mode)
    out_path = out_dir / f"{Path(csv_path).stem}_{mode}_permuted_seed{seed}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run row/column permutation stability checks for PolyTab.")
    parser.add_argument("--task", default="md")
    parser.add_argument("--keep-mode", default="keep25")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/permutation_stability")
    args = parser.parse_args()
    cfg = resolve_task_config(args.task)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [("original", Path(cfg["csv_path"])),
                ("row_permuted", make_variant(cfg["csv_path"], cfg["smiles_col"], "row", args.seed, out_dir)),
                ("column_permuted", make_variant(cfg["csv_path"], cfg["smiles_col"], "column", args.seed, out_dir))]
    runs = {}
    for label, csv_path in variants:
        predictor = create_predictor()
        results, _ = predictor.run_pipeline(str(csv_path), cfg["smiles_col"], args.keep_mode, f"{args.task}_{label}")
        runs[label] = summarize(results)
    summary_path = out_dir / f"{args.task}_{args.keep_mode}_permutation_summary.json"
    summary_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(runs, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
