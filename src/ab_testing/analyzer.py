"""
Statistical significance analysis for A/B testing.
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import ttest_ind


@dataclass
class SignificanceResult:
    p_value: float
    is_significant: bool
    winner: Optional[str]
    recommendation: str


class ABTestAnalyzer:
    """
    Performs statistical significance testing using prediction logs.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def analyze_significance(self, experiment_id: str) -> SignificanceResult:
        """
        Performs Welch's t-test on absolute prediction error.
        """
        log_file = f"logs/{experiment_id}_predictions.jsonl"

        if not os.path.exists(log_file):
            raise FileNotFoundError("Prediction log file not found")

        control_errors = []
        treatment_errors = []

        # Read logs
        with open(log_file, "r") as f:
            for line in f:
                record = json.loads(line)
                error = abs(record["error"])

                if record["variant"] == "control":
                    control_errors.append(error)
                elif record["variant"] == "treatment":
                    treatment_errors.append(error)

        if len(control_errors) < 2 or len(treatment_errors) < 2:
            raise ValueError(
                f"Need at least 2 samples per variant "
                f"(control={len(control_errors)}, treatment={len(treatment_errors)})"
            )

        control_mean = np.mean(control_errors)
        treatment_mean = np.mean(treatment_errors)

        # Welch's t-test (unequal variance)
        _, p_value = ttest_ind(
            control_errors,
            treatment_errors,
            equal_var=False
        )

        is_significant = bool(p_value < self.alpha)

        if is_significant:
            if treatment_mean < control_mean:
                winner = "treatment"
                recommendation = (
                    f"Treatment significantly better (p={p_value:.4f}). "
                    "Promote treatment model."
                )
            else:
                winner = "control"
                recommendation = (
                    f"Control significantly better (p={p_value:.4f}). "
                    "Keep current model."
                )
        else:
            winner = None
            recommendation = (
                f"No statistically significant difference (p={p_value:.4f}). "
                "Collect more data."
            )

        return SignificanceResult(
            p_value=round(p_value, 4),
            is_significant=is_significant,
            winner=winner,
            recommendation=recommendation
        )
