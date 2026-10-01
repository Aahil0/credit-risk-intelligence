import numpy as np
import pandas as pd
import pytest
from src.config import RAW_FEATURES, STATUS, BILLS, PAYMENTS
from src.features.engineering import engineer, preprocessing


def record():
    return {c: 10000 if c == 'LIMIT_BAL' else 0 for c in RAW_FEATURES}


def test_ratios_zero_negative_bills_and_delays():
    row = record()
    row.update({'BILL_AMT1': 5000, 'BILL_AMT2': -100, 'PAY_AMT1': 1000, 'PAY_AMT2': 2000,
                'PAY_0': 2, 'PAY_2': 1, 'PAY_3': -1})
    out = engineer(pd.DataFrame([row]))
    assert out.utilization_recent.iloc[0] == .5
    assert out.payment_bill_ratio_recent.iloc[0] == .2
    assert out.delinquency_frequency.iloc[0] == 2
    assert out.severe_delinquency_months.iloc[0] == 1
    assert out.nonpositive_bill_months.iloc[0] == 5
    assert np.isfinite(out.to_numpy()).all()


def test_trend_is_chronological():
    row = record()
    row.update(dict(zip(BILLS, [600,500,400,300,200,100])))
    row.update(dict(zip(PAYMENTS, [100,200,300,400,500,600])))
    out = engineer(pd.DataFrame([row]))
    assert out.bill_trend_per_limit.iloc[0] == pytest.approx(.01)
    assert out.payment_trend_per_limit.iloc[0] == pytest.approx(-.01)


def test_id_demographics_and_target_never_become_features():
    row = record()
    X = pd.DataFrame([{**row, 'ID': 1, 'SEX': 1, 'AGE': 20, 'default payment next month': 0},
                      {**row, 'ID': 500, 'SEX': 2, 'AGE': 70, 'default payment next month': 1}])
    prep = preprocessing().fit(X)
    np.testing.assert_allclose(prep.transform(X)[0], prep.transform(X)[1])
    assert not set(['ID','SEX','AGE','default payment next month']) & set(prep.get_feature_names_out())


def test_nonfinite_missing_and_zero_limit_rejected():
    with pytest.raises(ValueError):
        engineer(pd.DataFrame([{}]))
    for invalid in [0, np.nan, np.inf]:
        row = record(); row['LIMIT_BAL'] = invalid
        with pytest.raises(ValueError):
            engineer(pd.DataFrame([row]))
