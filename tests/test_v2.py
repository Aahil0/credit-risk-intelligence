import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from app.schemas import Applicant, Prediction, RiskFactor
from pydantic import ValidationError
from src.config import ROOT, RAW_FEATURES
from src.evaluation.fairness import wilson, fairness_audit
from src.evaluation.segmentation import define_segments, review_flag
from src.features.engineering import engineer
from src.models.service import RiskService, risk_category


def test_segments_and_review_are_independent_at_boundaries():
    cuts = define_segments([.0, .1, .2, .3, .4])
    assert cuts == {'low_upper_exclusive': .1, 'high_lower_inclusive': .3}
    assert risk_category(np.nextafter(.1, 0), cuts) == 'LOW'
    assert risk_category(.1, cuts) == 'MEDIUM'
    assert risk_category(np.nextafter(.3, 0), cuts) == 'MEDIUM'
    assert risk_category(.3, cuts) == 'HIGH'
    assert not review_flag(np.nextafter(.15, 0), .15)
    assert review_flag(.15, .15)
    assert risk_category(.15, cuts) == 'MEDIUM'


@pytest.mark.parametrize('values', [[], [.1, .1], [np.nan, .1], [-.1, .5], [.1, 1.1]])
def test_invalid_segment_evidence(values):
    with pytest.raises(ValueError):
        define_segments(values)


@pytest.mark.parametrize('field,value', [('PAY_0', -3), ('PAY_0', 10), ('PAY_0', .5), ('PAY_AMT1', -1)])
def test_engineering_rejects_invalid_financial_history(field, value):
    record = {c: 10000 if c == 'LIMIT_BAL' else 0 for c in RAW_FEATURES}
    record[field] = value
    with pytest.raises(ValueError):
        engineer(pd.DataFrame([record]))


def test_status_domain_and_zero_payment_behavior():
    record = {c: 10000 if c == 'LIMIT_BAL' else 0 for c in RAW_FEATURES}
    record.update(PAY_0=-2, PAY_2=9, BILL_AMT1=-100)
    result = engineer(pd.DataFrame([record])).iloc[0]
    assert result.zero_payment_months == 6
    assert result.payment_bill_ratio_mean == 0
    assert result.utilization_recent == -.01
    assert result.max_delay == 9


def test_wilson_interval_and_small_denominator_suppression():
    lower, upper = wilson(50, 100)
    assert lower == pytest.approx(.40383153)
    assert upper == pytest.approx(.59616847)
    assert wilson(0, 20) == (None, None)
    data = pd.DataFrame({'SEX': [1]*100, 'EDUCATION': [2]*100, 'MARRIAGE': [1]*100, 'AGE': [30]*100})
    row = fairness_audit(data, [1]*20+[0]*80, [.2]*100, .15).iloc[0]
    assert pd.isna(row.tpr) and pd.isna(row.tpr_ci95_lower)
    assert row.fpr == 1 and row.flag_rate == 1 and row.default_rate == .2
    assert row.calibration_gap == pytest.approx(0)


def test_missing_artifact_and_checksum_fail_before_deserialization(tmp_path, monkeypatch):
    missing = tmp_path/'missing.joblib'
    with pytest.raises(FileNotFoundError):
        RiskService(missing)
    missing.write_bytes(b'untrusted')
    missing.with_suffix('.sha256').write_text('incorrect')
    monkeypatch.setattr('src.models.service.joblib.load', lambda *args: pytest.fail('Must not deserialize'))
    with pytest.raises(ValueError, match='checksum'):
        RiskService(missing)


@pytest.mark.parametrize('change', [{'credit_limit': True}, {'repayment_status': [False]*6},
    {'payment_amounts': [None]*6}, {'bill_amounts': [float('inf')]*6}])
def test_strict_inputs(change):
    payload = {'credit_limit': 200000, 'repayment_status': [0]*6, 'bill_amounts': [0]*6, 'payment_amounts': [0]*6}
    with pytest.raises(ValidationError):
        Applicant(**{**payload, **change})


@pytest.mark.skipif(not (ROOT/'models/credit_risk.joblib').exists(), reason='Artifact unavailable')
def test_readable_explanation_values_and_frozen_metadata():
    service = RiskService()
    applicant = Applicant(credit_limit=200000, repayment_status=[0]*6, bill_amounts=[45000]*6, payment_amounts=[5000]*6)
    result = service.predict(applicant)
    Prediction.model_validate(result)
    engineered = engineer(pd.DataFrame([{'LIMIT_BAL': 200000, **dict(zip(['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], [0]*6)),
        **{f'BILL_AMT{i}': 45000 for i in range(1,7)}, **{f'PAY_AMT{i}': 5000 for i in range(1,7)}}])).iloc[0]
    for factor in result['important_risk_factors']:
        RiskFactor.model_validate(factor)
        if factor['feature'] in engineered:
            assert factor['feature_value'] == engineered[factor['feature']]
        else:
            assert factor['value_unit'] == 'repayment code'
            assert factor['feature_value'] == 0
            assert factor['indicator_active'] == (factor['indicator_code'] == 0)
    metadata = json.loads((ROOT/'models/metadata.json').read_text())
    assert service.metadata == metadata
    frozen = json.loads((ROOT/'reports/frozen_decisions.json').read_text())
    assert all(metadata[k] == value for k, value in frozen.items())
    assert metadata['segments']['high_lower_inclusive'] != metadata['threshold']
