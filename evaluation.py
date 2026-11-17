"""Evaluation helpers framed around the business decision: whom to call."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)


def score_table(y_true, proba_by_model: dict, threshold: float = 0.5) -> pd.DataFrame:
    """One row per model with ranking, calibration and classification metrics."""
    rows = []
    for name, p in proba_by_model.items():
        rows.append({
            "model": name,
            "ROC-AUC": roc_auc_score(y_true, p),
            "PR-AUC": average_precision_score(y_true, p),
            "Brier": brier_score_loss(y_true, p),
            f"F1@{threshold}": f1_score(y_true, (p >= threshold).astype(int)),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def gains_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gains / lift by decile of predicted probability.

    Reads as: 'if we only call the top k% of customers ranked by the model,
    what share of all subscribers do we reach, and how much better is that
    than calling at random?'
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    g = df.groupby("decile").agg(customers=("y", "size"), subscribers=("y", "sum"))
    g["pct_customers_called"] = g["customers"].cumsum() / g["customers"].sum()
    g["pct_subscribers_reached"] = g["subscribers"].cumsum() / g["subscribers"].sum()
    g["conversion_rate"] = g["subscribers"] / g["customers"]
    g["cumulative_lift"] = g["pct_subscribers_reached"] / g["pct_customers_called"]
    return g


def profit_curve(y_true, proba, cost_per_call: float, value_per_deposit: float):
    """Expected campaign profit when calling everyone above each threshold.

    Cost/value figures are ASSUMPTIONS passed in by the user, not data.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba)
    thresholds = np.linspace(0, 1, 101)
    profits = []
    for t in thresholds:
        called = p >= t
        profits.append(y[called].sum() * value_per_deposit - called.sum() * cost_per_call)
    return pd.DataFrame({"threshold": thresholds, "profit": profits})
