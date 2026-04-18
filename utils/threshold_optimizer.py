"""Systematic threshold selection utilities.

This module centralises threshold search so training scripts can replace
hardcoded probability cutoffs with a reproducible, validation-driven
procedure. Optuna is used when available; otherwise a deterministic grid
search fallback keeps the pipeline runnable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import optuna
except ImportError:  # pragma: no cover - fallback for minimal environments
    optuna = None


def _to_numpy(array_like) -> np.ndarray:
    values = np.asarray(array_like, dtype=float)
    if values.ndim != 1:
        return values.reshape(-1)
    return values


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def compute_threshold_metrics(y_true, y_prob, threshold: float) -> dict:
    """Compute binary classification metrics at a fixed threshold."""
    y_true_arr = _to_numpy(y_true).astype(int)
    y_prob_arr = _to_numpy(y_prob)
    y_pred_arr = (y_prob_arr >= float(threshold)).astype(int)

    counts = _confusion_counts(y_true_arr, y_pred_arr)
    tp = counts['tp']
    tn = counts['tn']
    fp = counts['fp']
    fn = counts['fn']

    recall = recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true_arr, y_pred_arr)

    youden_j = float(recall + specificity - 1.0)
    f_ss = float(0.0 if (recall + specificity) == 0 else (2.0 * recall * specificity) / (recall + specificity))

    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'youden_j': float(youden_j),
        'f_ss': float(f_ss),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def optimize_threshold(y_true,
                      y_prob,
                      objective: str = 'youden_j',
                      n_trials: int = 200,
                      seed: int = 42,
                      low: float = 0.01,
                      high: float = 0.99,
                      timeout: Optional[float] = None,
                      min_recall: Optional[float] = None) -> tuple[float, dict, object | None]:
    """Optimize a binary decision threshold.

    Parameters
    ----------
    objective:
        Either 'youden_j' or 'f_ss'.
    min_recall:
        Optional recall floor. If provided, thresholds below the floor are
        penalised during search.
    """
    objective = objective.lower().strip()
    if objective not in {'youden_j', 'f_ss'}:
        raise ValueError("objective must be 'youden_j' or 'f_ss'")

    y_true_arr = _to_numpy(y_true).astype(int)
    y_prob_arr = _to_numpy(y_prob)

    if len(y_true_arr) == 0:
        raise ValueError('y_true must contain at least one sample')

    def score_threshold(threshold: float) -> tuple[float, dict]:
        metrics = compute_threshold_metrics(y_true_arr, y_prob_arr, threshold)
        score = float(metrics[objective])
        if min_recall is not None and metrics['recall'] < min_recall:
            score -= float((min_recall - metrics['recall']) * 10.0)
        return score, metrics

    if optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        def _objective(trial):
            threshold = trial.suggest_float('threshold', low, high)
            score, metrics = score_threshold(threshold)
            trial.set_user_attr('metrics', metrics)
            return score

        study.optimize(_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best_threshold = float(study.best_params['threshold'])
        best_metrics = dict(study.best_trial.user_attrs.get('metrics') or compute_threshold_metrics(y_true_arr, y_prob_arr, best_threshold))
        best_metrics['objective'] = objective
        best_metrics['optuna_best_value'] = float(study.best_value)
        best_metrics['optuna_trials'] = int(len(study.trials))
        best_metrics['search_method'] = 'optuna'
        return best_threshold, best_metrics, study

    thresholds = np.linspace(low, high, max(n_trials, 50))
    best_threshold = float(thresholds[0])
    best_metrics = compute_threshold_metrics(y_true_arr, y_prob_arr, best_threshold)
    best_score = float(best_metrics[objective])

    for threshold in thresholds[1:]:
        score, metrics = score_threshold(float(threshold))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics = dict(best_metrics)
    best_metrics['objective'] = objective
    best_metrics['optuna_best_value'] = float(best_score)
    best_metrics['optuna_trials'] = int(len(thresholds))
    best_metrics['search_method'] = 'grid_fallback'
    return best_threshold, best_metrics, None
