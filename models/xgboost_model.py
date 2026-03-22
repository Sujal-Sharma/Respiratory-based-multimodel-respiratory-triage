"""
models/xgboost_model.py — XGBoost classifier configuration for COUGHVID metadata.
"""

from xgboost import XGBClassifier
from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, COUGHVID_CLASSES
)


def build_xgboost(num_classes: int = None) -> XGBClassifier:
    """
    Build and return a configured XGBClassifier.

    Uses GPU (device='cuda') with histogram method for fast training.
    Early stopping must be set during model.fit() call.

    Parameters
    ----------
    num_classes : int  — number of target classes (default: len(COUGHVID_CLASSES))

    Returns
    -------
    model : XGBClassifier (untrained)
    """
    if num_classes is None:
        num_classes = len(COUGHVID_CLASSES)

    # Binary classification (2 classes: Healthy vs Symptomatic)
    # use binary:logistic; multi:softprob requires num_class≥3
    if num_classes == 2:
        objective  = 'binary:logistic'
        eval_metric = 'logloss'
        extra_kw    = {}
    else:
        objective  = 'multi:softprob'
        eval_metric = 'mlogloss'
        extra_kw    = {'num_class': num_classes}

    model = XGBClassifier(
        n_estimators          = XGB_N_ESTIMATORS,
        max_depth             = XGB_MAX_DEPTH,
        learning_rate         = XGB_LEARNING_RATE,
        subsample             = 0.85,
        colsample_bytree      = 0.85,
        min_child_weight      = 3,
        gamma                 = 0.1,
        reg_alpha             = 0.05,
        reg_lambda            = 1.0,
        eval_metric           = eval_metric,
        early_stopping_rounds = 20,
        tree_method           = 'hist',
        device                = 'cuda',
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 1,
        objective             = objective,
        **extra_kw,
    )

    print(f"[xgboost_model] XGBClassifier built for {num_classes} classes")
    print(f"  n_estimators = {XGB_N_ESTIMATORS}")
    print(f"  max_depth    = {XGB_MAX_DEPTH}")
    print(f"  learning_rate= {XGB_LEARNING_RATE}")
    print(f"  device       = cuda (GPU)")
    return model


if __name__ == "__main__":
    model = build_xgboost()
    print(model)
