"""Stateless historical features: no fitting, target or future information."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.config import RAW_FEATURES, STATUS, BILLS, PAYMENTS


def engineer(X):
    missing = set(RAW_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    out = X[RAW_FEATURES].astype(float).copy()
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Features must be finite')
    bills = out[BILLS].to_numpy()
    payments = out[PAYMENTS].to_numpy()
    status = out[STATUS].to_numpy()
    limit = out.LIMIT_BAL.to_numpy()
    if (limit <= 0).any():
        raise ValueError('LIMIT_BAL must be positive')
    if (payments < 0).any():
        raise ValueError('Payments must be nonnegative')
    if not np.isin(status, np.arange(-2, 10)).all():
        raise ValueError('Repayment statuses must be whole codes from -2 to 9')
    # Negative statements represent credits; retain signed balances, use positive
    # balances only as ratio denominators, plus flags for nonpositive statements.
    out['utilization_recent'] = bills[:, 0] / limit
    out['utilization_mean'] = bills.mean(axis=1) / limit
    out['utilization_max'] = bills.max(axis=1) / limit
    out['over_limit_months'] = (bills > limit[:, None]).sum(axis=1)
    out['nonpositive_bill_months'] = (bills <= 0).sum(axis=1)
    ratios = np.divide(payments, bills, out=np.zeros_like(payments), where=bills > 0)
    out['payment_bill_ratio_recent'] = np.clip(ratios[:, 0], 0, 10)
    out['payment_bill_ratio_mean'] = np.clip(ratios, 0, 10).mean(axis=1)
    out['zero_payment_months'] = (payments == 0).sum(axis=1)
    out['payment_limit_mean'] = payments.mean(axis=1) / limit
    out['delinquency_frequency'] = (status > 0).sum(axis=1)
    out['severe_delinquency_months'] = (status >= 2).sum(axis=1)
    out['max_delay'] = np.maximum(status, 0).max(axis=1)
    out['recent_delay_change'] = np.maximum(status[:, 0], 0) - np.maximum(status[:, 1], 0)
    # Arrays are September -> April. Reverse to obtain chronological slopes.
    centered_time = np.arange(6) - 2.5
    out['bill_trend_per_limit'] = (bills[:, ::-1] @ centered_time) / (17.5 * limit)
    out['payment_trend_per_limit'] = (payments[:, ::-1] @ centered_time) / (17.5 * limit)
    return out


ENGINEERED_FEATURES = list(engineer(pd.DataFrame([{c: 10000 if c == 'LIMIT_BAL' else 0 for c in RAW_FEATURES}])).columns)
NUMERIC = [c for c in ENGINEERED_FEATURES if c not in STATUS]


class CreditFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        engineer(X)
        self.n_features_in_ = len(RAW_FEATURES)
        return self
    def transform(self, X):
        return engineer(X)
    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED_FEATURES)


def preprocessing():
    return Pipeline([
        ('features', CreditFeatures()),
        ('columns', ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC),
            ('status', OneHotEncoder(categories=[list(range(-2, 10))]*6,
                                     handle_unknown='error', sparse_output=False), STATUS)
        ], verbose_feature_names_out=False))
    ])
