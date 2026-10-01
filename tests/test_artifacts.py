import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from src.config import ROOT, RAW_FEATURES, STATUS, BILLS, PAYMENTS
from src.evaluation.metrics import evaluate
from src.models.service import RiskService, risk_category
from src.models.explain import explain
from app.schemas import Applicant


def test_segmentation_boundaries():
    cuts = {'low_upper_exclusive': .075, 'high_lower_inclusive': .15}
    assert risk_category(.0749, cuts) == 'LOW'
    assert risk_category(.075, cuts) == 'MEDIUM'
    assert risk_category(.15, cuts) == 'HIGH'


def test_known_metric_counts():
    metrics = evaluate([0,0,1,1], [.1,.7,.2,.8], .5)
    assert [metrics[k] for k in ['tn','fp','fn','tp']] == [1,1,1,1]
    assert metrics['cost_per_customer'] == 1.5


requires_artifact = pytest.mark.skipif(not (ROOT/'models/credit_risk.joblib').exists(), reason='Training artifact not available')


@requires_artifact
def test_artifact_checksum_and_recomputed_reported_metrics():
    path = ROOT/'models/credit_risk.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix('.sha256').read_text().strip()
    metadata = json.loads((ROOT/'models/metadata.json').read_text())
    predictions = pd.read_csv(ROOT/'reports/test_predictions.csv')
    actual = evaluate(predictions.y_true, predictions[metadata['selected_model']], metadata['threshold'])
    for key, value in actual.items():
        assert value == pytest.approx(metadata['test_metrics'][key], abs=1e-10)
    table = pd.read_csv(ROOT/'reports/test_model_comparison.csv')
    final = table[table.model == metadata['selected_model']].iloc[0]
    for key in actual:
        assert final[key] == pytest.approx(actual[key])


@requires_artifact
def test_groups_disjoint_and_test_not_in_selection():
    manifest = pd.read_csv(ROOT/'reports/split_manifest.csv', dtype={'group': str})
    assert manifest.groupby('group')['split'].nunique().max() == 1
    assert manifest.ID.nunique() == 30000
    assert set(manifest.split) == {'train','calibration','validation','test'}
    frozen = json.loads((ROOT/'reports/frozen_decisions.json').read_text())
    rows = pd.read_csv(ROOT/'reports/v2_selection_audit.csv')
    assert rows.loc[rows.raw_validation_ap.idxmax(), 'model'] == frozen['selected_model']
    thresholds = pd.read_csv(ROOT/'reports/validation_thresholds.csv')
    selected_cost = thresholds.loc[np.isclose(thresholds.threshold, frozen['threshold']), 'cost_per_customer'].iloc[0]
    assert selected_cost == pytest.approx(thresholds.cost_per_customer.min())


@requires_artifact
def test_shap_additivity_and_calibrated_prediction():
    service = RiskService()
    profiles = json.loads((ROOT/'app/demo_profiles.json').read_text())
    for profile in profiles.values():
        X = pd.DataFrame([profile], columns=RAW_FEATURES)
        explanation = explain(service.raw, service.explainer, X)
        model = service.raw.named_steps['model']
        transformed = service.raw.named_steps['prep'].transform(X)
        total = float(explanation.base_values[0] + explanation.values[0].sum())
        if model.__class__.__name__ == 'RandomForestClassifier':
            expected = float(model.predict_proba(transformed)[0,1])
        elif model.__class__.__name__ == 'LogisticRegression':
            expected = float(model.decision_function(transformed)[0])
        else:
            expected = float(model.predict(transformed, output_margin=True)[0])
        assert total == pytest.approx(expected, abs=1e-5)
        applicant = Applicant(credit_limit=profile['LIMIT_BAL'], repayment_status=[profile[c] for c in STATUS],
                              bill_amounts=[profile[c] for c in BILLS], payment_amounts=[profile[c] for c in PAYMENTS])
        result = service.predict(applicant)
        assert result['probability_of_default'] == pytest.approx(service.predictor.predict_proba(X)[0,1])
        assert result['risk_category'] == risk_category(result['probability_of_default'], service.metadata['segments'])


@requires_artifact
def test_corrupt_artifact_refused(tmp_path):
    bad = tmp_path/'bad.joblib'; bad.write_bytes(b'corrupted')
    bad.with_suffix('.sha256').write_text('invalid')
    with pytest.raises(ValueError, match='checksum'):
        RiskService(bad)


@requires_artifact
def test_outside_training_support_is_flagged():
    service = RiskService()
    applicant = Applicant(credit_limit=10_000_000, repayment_status=[0]*6,
                          bill_amounts=[0]*6, payment_amounts=[0]*6)
    result = service.predict(applicant)
    assert any('LIMIT_BAL' in w for w in result['warnings'])
    assert 0 <= result['probability_of_default'] <= 1
