"""Feature engineering and preprocessing for the bank term-deposit model.

Kept in a module (not inside the notebook) so that a saved model can be
re-loaded with joblib from any script: joblib stores a *reference* to
`add_contact_features`, so the function must be importable.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

TARGET = "deposit"

# `duration` (length of the last call) is only known AFTER the call ends,
# so it cannot be used to decide whom to call. It is excluded by default.
LEAKY_COLS = ["duration"]

NUMERIC_COLS = ["age", "balance", "day", "campaign", "pdays", "previous"]
ORDINAL_COLS = ["education"]
EDUCATION_ORDER = [["unknown", "primary", "secondary", "tertiary"]]
NOMINAL_COLS = ["job", "marital", "default", "housing", "loan",
                "contact", "month", "poutcome", "contacted_before"]


def add_contact_features(X: pd.DataFrame) -> pd.DataFrame:
    """`pdays = -1` means 'never contacted before', not a number of days.

    Split it into a binary flag + a numeric column where -1 becomes NaN
    (later imputed), so the model does not read -1 as 'contacted 1 day ago'.
    """
    X = X.copy()
    obj_cols = X.select_dtypes(include="object").columns
    X[obj_cols] = X[obj_cols].where(X[obj_cols].notna(), np.nan)
    X["contacted_before"] = (X["pdays"] != -1).map({True: "yes", False: "no"})
    X["pdays"] = X["pdays"].replace(-1, np.nan)
    return X


def make_preprocessor(scale_numeric: bool = True,
                      include_duration: bool = False) -> ColumnTransformer:
    """Column-wise preprocessing.

    scale_numeric=True for distance/linear models (KNN, logistic regression);
    tree models are scale-invariant, so scaling is skipped for them.
    """
    num_cols = NUMERIC_COLS + (LEAKY_COLS if include_duration else [])
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    cat_impute = SimpleImputer(strategy="constant", fill_value="unknown")
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("ord", Pipeline([("impute", cat_impute),
                          ("encode", OrdinalEncoder(categories=EDUCATION_ORDER))]),
         ORDINAL_COLS),
        ("cat", Pipeline([("impute", cat_impute),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]),
         NOMINAL_COLS),
    ], verbose_feature_names_out=False)


def make_pipeline(model, scale_numeric: bool = True,
                  include_duration: bool = False) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> model."""
    return Pipeline([
        ("features", FunctionTransformer(add_contact_features)),
        ("preprocess", make_preprocessor(scale_numeric, include_duration)),
        ("model", model),
    ])


def load_data(path: str) -> pd.DataFrame:
    """Load either the course pickle."""
    df = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
    # The public CSV codes missing values as the string "unknown"; normalise to NaN
    # so both sources go through the same imputation step.
    for col in ["job", "education"]:
        df[col] = df[col].replace("unknown", np.nan)
    return df
