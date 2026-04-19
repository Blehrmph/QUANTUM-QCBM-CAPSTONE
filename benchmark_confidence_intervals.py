"""
benchmark_confidence_intervals.py — Stage 1 QCBM stability across 5 random seeds.

Runs main.py --stage1-only for seeds 0, 42, 123, 256, 999 and reports
mean ± std on all key metrics using the LR+Isotonic operating point.
Results saved to artifacts/confidence_intervals.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SEEDS = [0, 42, 123, 256, 999]
CONFIG = "best_config.json"
OUT_BASE = Path("artifacts/ci_runs")

METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall_dr", "far", "mcc"]
METRIC_LABELS = {
    "roc_auc":    "ROC-AUC",
    "pr_auc":     "PR-AUC",
    "f1":         "F1",
    "precision":  "Precision",
    "recall_dr":  "Recall/DR",
    "far":        "FAR",
    "mcc":        "MCC",
}


def run_seed(seed: int) -> dict:
    out_dir = OUT_BASE / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "main.py",
        "--config", CONFIG,
        "--output-dir", str(out_dir),
        "--stage1-only",
        "--seed", str(seed),
    ]
    print(f"\n[CI] Running seed={seed} ...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [CI] seed={seed} FAILED (exit {result.returncode})")
        return {}
    metrics_path = out_dir / "hier_stage1_metrics.json"
    if not metrics_path.exists():
        print(f"  [CI] seed={seed}: metrics file not found")
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def extract(metrics: dict) -> dict:
    """Pull the LR+Isotonic (best) operating point metrics."""
    iso = metrics.get("isotonic_calibration_metrics", {})
    return {k: iso.get(k, float("nan")) for k in METRICS}


def main():
    all_results = {}
    extracted = []

    for seed in SEEDS:
        m = run_seed(seed)
        all_results[seed] = m
        if m:
            extracted.append(extract(m))

    if not extracted:
        print("No successful runs — cannot compute confidence intervals.")
        return

    print("\n" + "=" * 72)
    print("  STAGE 1 CONFIDENCE INTERVALS  (LR + Isotonic operating point)")
    print(f"  Seeds: {SEEDS}  |  n={len(extracted)} successful runs")
    print("=" * 72)
    print(f"  {'Metric':<14}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'95% CI':>20}")
    print("  " + "-" * 70)

    ci_results = {}
    for k in METRICS:
        vals = np.array([e[k] for e in extracted if not np.isnan(e[k])])
        if len(vals) == 0:
            continue
        mean, std = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        lo, hi = mean - 1.96 * std, mean + 1.96 * std
        label = METRIC_LABELS[k]
        print(f"  {label:<14}  {mean:>8.4f}  {std:>8.4f}  {vals.min():>8.4f}  {vals.max():>8.4f}  [{lo:.4f}, {hi:.4f}]")
        ci_results[k] = {
            "mean": round(float(mean), 6),
            "std":  round(float(std), 6),
            "min":  round(float(vals.min()), 6),
            "max":  round(float(vals.max()), 6),
            "ci_95_lo": round(float(lo), 6),
            "ci_95_hi": round(float(hi), 6),
            "values": [round(float(v), 6) for v in vals.tolist()],
        }

    print("=" * 72)

    out = {
        "seeds": SEEDS,
        "n_successful": len(extracted),
        "operating_point": "LR+Isotonic",
        "metrics": ci_results,
        "per_seed_raw": {str(s): all_results[s] for s in SEEDS},
    }
    out_path = Path("artifacts/confidence_intervals.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
