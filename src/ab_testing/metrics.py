"""
Metrics computation for A/B testing.
Reads prediction logs and compares control vs treatment.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_predictions(experiment_id: str, log_dir: str = "logs"):
    """
    Load predictions from JSONL log file.
    """
    log_file = Path(log_dir) / f"{experiment_id}_predictions.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"No logs found for experiment {experiment_id}")

    records = []
    with open(log_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    """
    return {
        "sample_size": len(y_true),
        "r2_score": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def calculate_ab_metrics(experiment_id: str):
    """
    Calculate metrics for control and treatment variants.
    """
    logs = load_predictions(experiment_id)

    grouped = {"control": [], "treatment": []}

    for row in logs:
        grouped[row["variant"]].append(row)
    
    # print(grouped)

    metrics = {}

    for variant, rows in grouped.items():
        if not rows:
            continue

        y_true = [r["ground_truth"] for r in rows]
        y_pred = [r["prediction"] for r in rows]

        metrics[variant] = compute_metrics(y_true, y_pred)

    return metrics


def compare_variants(metrics: dict):
    """
    Compare treatment against control.
    """
    if "control" not in metrics or "treatment" not in metrics:
        return {"error": "Both control and treatment required"}

    control = metrics["control"]
    treatment = metrics["treatment"]

    def improvement(ctrl, trt, lower_is_better=False):
        if ctrl == 0:
            return 0.0
        diff_pct = ((trt - ctrl) / abs(ctrl)) * 100
        return round(-diff_pct if lower_is_better else diff_pct, 2)

    return {
        "control": control,
        "treatment": treatment,
        "comparison": {
            "r2_improvement_%": improvement(control["r2_score"], treatment["r2_score"]),
            "mae_improvement_%": improvement(control["mae"], treatment["mae"], lower_is_better=True),
            "rmse_improvement_%": improvement(control["rmse"], treatment["rmse"], lower_is_better=True),
        },
    }

def save_metrics(comparison: dict, experiment_id: str, out_dir="evaluation"):
    Path(out_dir).mkdir(exist_ok=True)
    out_file = Path(out_dir) / f"{experiment_id}_metrics.json"
    with open(out_file, "w") as f:
        json.dump(comparison, f, indent=2)

