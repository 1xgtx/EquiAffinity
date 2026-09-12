"""Scientific evaluation metrics for protein--ligand affinity prediction.

All metrics are calculated after pairwise removal of NaN and infinite values.
This prevents a single failed prediction from silently contaminating an entire
validation result while ensuring the target/prediction alignment is preserved.
Pearson's R is mathematically undefined for a constant vector; in that case
``compute_pearson_r`` returns ``nan`` rather than raising or reporting a
misleading correlation of zero.

Run the deterministic self-test with:

    python -m src.utils.metrics
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:  # NumPy and PyTorch are optional for lightweight result-only environments.
    import numpy as np
except ImportError:  # pragma: no cover - depends on the local environment
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None


def _as_finite_pairs(y_true: Any, y_pred: Any) -> tuple[List[float], List[float]]:
    """Flatten supported inputs and return paired finite observations only."""
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().reshape(-1).tolist()
    elif np is not None and isinstance(y_true, np.ndarray):
        y_true = y_true.reshape(-1).tolist()
    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().reshape(-1).tolist()
    elif np is not None and isinstance(y_pred, np.ndarray):
        y_pred = y_pred.reshape(-1).tolist()

    true_values = [float(value) for value in y_true]
    pred_values = [float(value) for value in y_pred]
    if len(true_values) != len(pred_values):
        raise ValueError(f"y_true and y_pred must have equal length, got {len(true_values)} and {len(pred_values)}.")
    filtered = [(target, prediction) for target, prediction in zip(true_values, pred_values)
                if math.isfinite(target) and math.isfinite(prediction)]
    if not filtered:
        raise ValueError("No paired finite values remain after removing NaN/Inf observations.")
    return [pair[0] for pair in filtered], [pair[1] for pair in filtered]


def compute_rmse(y_true: Any, y_pred: Any) -> float:
    """Compute RMSE after excluding paired NaN/Inf values.

    Accepts ``torch.Tensor``, ``numpy.ndarray``, or any one-dimensional
    numerical iterable. At least one finite target/prediction pair is required.
    """
    targets, predictions = _as_finite_pairs(y_true, y_pred)
    squared_error_sum = math.fsum((prediction - target) ** 2 for target, prediction in zip(targets, predictions))
    return math.sqrt(squared_error_sum / len(targets))


def compute_pearson_r(y_true: Any, y_pred: Any) -> float:
    """Compute numerically stable Pearson R; return NaN for zero variance.

    The centred two-pass form and ``math.fsum`` reduce cancellation error for
    affinity labels with a narrow dynamic range.
    """
    targets, predictions = _as_finite_pairs(y_true, y_pred)
    if len(targets) < 2:
        return float("nan")
    mean_target = math.fsum(targets) / len(targets)
    mean_prediction = math.fsum(predictions) / len(predictions)
    target_delta = [value - mean_target for value in targets]
    prediction_delta = [value - mean_prediction for value in predictions]
    numerator = math.fsum(left * right for left, right in zip(target_delta, prediction_delta))
    target_sum_squares = math.fsum(value * value for value in target_delta)
    prediction_sum_squares = math.fsum(value * value for value in prediction_delta)
    denominator = math.sqrt(target_sum_squares * prediction_sum_squares)
    # The scale-aware guard avoids instability for effectively constant inputs.
    if denominator <= math.ulp(1.0) * max(target_sum_squares, prediction_sum_squares, 1.0):
        return float("nan")
    return max(-1.0, min(1.0, numerator / denominator))


def evaluate_affinity_predictions(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Return the standard affinity-prediction metrics in one serialisable dict."""
    return {"rmse": compute_rmse(y_true, y_pred), "pearson_r": compute_pearson_r(y_true, y_pred)}


def aggregate_multi_seed_metrics(seed_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float | str]]:
    """Aggregate at least three seed results as mean ± sample standard deviation.

    ``seed`` is treated as metadata and excluded. Every other metric must be
    numeric and finite in every seed result. The returned ``formatted`` fields
    are suitable for direct inclusion in a manuscript table.
    """
    if len(seed_results) < 3:
        raise ValueError("At least 3 random-seed results are required for sample standard deviation.")
    first_metrics = set(seed_results[0]) - {"seed"}
    if not first_metrics:
        raise ValueError("Each seed result must include at least one metric besides 'seed'.")
    summary: Dict[str, Dict[str, float | str]] = {}
    for metric in sorted(first_metrics):
        values = []
        for index, result in enumerate(seed_results):
            if set(result) - {"seed"} != first_metrics:
                raise ValueError(f"Seed result {index} has inconsistent metric keys.")
            value = float(result[metric])
            if not math.isfinite(value):
                raise ValueError(f"Seed result {index} has non-finite {metric}: {value!r}")
            values.append(value)
        mean = statistics.fmean(values)
        std = statistics.stdev(values)  # ddof=1: sample standard deviation.
        summary[metric] = {"mean": mean, "std": std, "formatted": f"{mean:.3f} ± {std:.3f}"}
    return summary


def _mock_core_set() -> tuple[Any, Any]:
    """Create a deterministic 290-complex Core-Set-like affinity benchmark."""
    targets = [4.0 + 6.0 * index / 289.0 for index in range(290)]
    predictions = [target + 0.32 * math.sin(0.37 * index) + 0.08 * math.cos(0.11 * index)
                   for index, target in enumerate(targets)]
    if torch is not None:
        return torch.tensor(targets, dtype=torch.float32), torch.tensor(predictions, dtype=torch.float32)
    if np is not None:
        return np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    return targets, predictions


def _print_baseline_table(seed_results: Sequence[Mapping[str, float]], summary: Mapping[str, Mapping[str, float | str]]) -> None:
    """Print an academic-report-ready, compact baseline table."""
    print("\n" + "=" * 68)
    print("EquiAffinity Core Set — Baseline Evaluation (290 simulated complexes)")
    print("=" * 68)
    print(f"{'Seed':<12}{'RMSE (pK)':>18}{'Pearson R':>18}")
    print("-" * 68)
    for result in seed_results:
        print(f"{int(result['seed']):<12}{result['rmse']:>18.3f}{result['pearson_r']:>18.3f}")
    print("-" * 68)
    print(f"{'Mean ± SD':<12}{summary['rmse']['formatted']:>18}{summary['pearson_r']['formatted']:>18}")
    print("=" * 68)


def run_self_test() -> None:
    """Validate metric correctness, edge-case handling, and multi-seed reporting."""
    y_true, y_pred = _mock_core_set()
    metrics = evaluate_affinity_predictions(y_true, y_pred)
    assert math.isfinite(metrics["rmse"]) and metrics["rmse"] > 0.0
    assert math.isfinite(metrics["pearson_r"]) and -1.0 <= metrics["pearson_r"] <= 1.0
    assert math.isclose(compute_rmse([1.0, float("nan"), 3.0], [2.0, 9.0, 5.0]), math.sqrt(2.5), rel_tol=1e-12)
    assert math.isnan(compute_pearson_r([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))

    seed_results = [
        {"seed": 42, "rmse": metrics["rmse"] + 0.012, "pearson_r": metrics["pearson_r"] - 0.004},
        {"seed": 123, "rmse": metrics["rmse"] - 0.008, "pearson_r": metrics["pearson_r"] + 0.003},
        {"seed": 456, "rmse": metrics["rmse"] + 0.004, "pearson_r": metrics["pearson_r"] + 0.001},
    ]
    summary = aggregate_multi_seed_metrics(seed_results)
    assert set(summary) == {"rmse", "pearson_r"}
    assert summary["rmse"]["std"] > 0.0 and "±" in summary["pearson_r"]["formatted"]
    _print_baseline_table(seed_results, summary)
    print("PASS: metric calculations, NaN handling, zero-variance guard, and 3-seed aggregation verified.\n")


if __name__ == "__main__":
    run_self_test()
