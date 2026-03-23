"""
models/xgboost_model.py — XGBoost classifier for COUGHVID cough classification.

Hyperparameters tuned per literature consensus:
  - Pahar et al. (2022): XGBoost on COUGHVID, 74-78% accuracy
  - Mohammed et al. (2023): Stacking ensemble, 78-82% accuracy
  - Lower LR (0.05) + more estimators (500) + deeper trees (8) for richer feature set
"""

from xgboost import XGBClassifier
from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, COUGHVID_CLASSES
)


def build_xgboost(num_classes: int = None,
                   scale_pos_weight: float = 1.0) -> XGBClassifier:
    """
    Build and return a configured XGBClassifier.

    Parameters
    ----------
    num_classes       : int   — number of target classes (default: len(COUGHVID_CLASSES))
    scale_pos_weight  : float — weight for positive class (handles imbalance)

    Returns
    -------
    model : XGBClassifier (untrained)
    """
    if num_classes is None:
        num_classes = len(COUGHVID_CLASSES)

    if num_classes == 2:
        objective   = 'binary:logistic'
        eval_metric = 'logloss'
        extra_kw    = {'scale_pos_weight': scale_pos_weight}
    else:
        objective   = 'multi:softprob'
        eval_metric = 'mlogloss'
        extra_kw    = {'num_class': num_classes}

    model = XGBClassifier(
        n_estimators          = XGB_N_ESTIMATORS,
        max_depth             = XGB_MAX_DEPTH,
        learning_rate         = XGB_LEARNING_RATE,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        colsample_bylevel     = 0.8,
        min_child_weight      = 5,
        gamma                 = 0.2,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
        eval_metric           = eval_metric,
        early_stopping_rounds = 30,
        tree_method           = 'hist',
        device                = 'cuda',
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 1,
        objective             = objective,
        **extra_kw,
    )

    print(f"[xgboost_model] XGBClassifier built for {num_classes} classes")
    print(f"  n_estimators = {XGB_N_ESTIMATORS}, max_depth = {XGB_MAX_DEPTH}")
    print(f"  learning_rate= {XGB_LEARNING_RATE}, scale_pos_weight= {scale_pos_weight}")
    print(f"  device       = cuda (GPU)")
    return model


if __name__ == "__main__":
    model = build_xgboost()
    print(model)
