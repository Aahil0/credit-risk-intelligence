"""Upgrade descriptive metadata without retraining or changing predictions.

Uses the existing validation partition for quartiles and selection audit.
Test predictions are read only after the new descriptive cuts are fixed.
V1 snapshot must be retained for provenance.
"""
import hashlib
import json
import zipfile
import joblib
import numpy as np
import pandas as pd
from src.config import ROOT, RAW_FEATURES, TARGET
from src.models.service import RiskService, risk_category
from src.evaluation.segmentation import define_segments
from src.evaluation.fairness import fairness_audit


def run():
    service = RiskService()
    if service.metadata.get('methodology_version') == '2.0.0':
        raise ValueError('V2 already applied; preserve the original V1 provenance')
    with zipfile.ZipFile(ROOT/'data/raw/uci.zip') as archive:
        with archive.open(next(n for n in archive.namelist() if n.endswith('.xls'))) as source:
            data = pd.read_excel(source, header=1).set_index('ID')
    manifest = pd.read_csv(ROOT/'reports/split_manifest.csv')
    validation = data.loc[manifest.loc[manifest.split == 'validation', 'ID']]
    probabilities = service.predictor.predict_proba(validation[RAW_FEATURES])[:, 1]
    segments = define_segments(probabilities)
    rows = []
    for path in sorted((ROOT/'reports').glob('calibration_*.csv')):
        table = pd.read_csv(path)
        raw = table.loc[table.calibration == 'none'].iloc[0]
        calibrated = table.loc[table.brier.idxmin()]
        rows.append({'model': raw.model, 'raw_validation_ap': raw.pr_auc_ap,
                     'raw_validation_roc_auc': raw.roc_auc, 'calibration': calibrated.calibration,
                     'selected_brier': calibrated.brier, 'selected_log_loss': calibrated.log_loss,
                     'calibrated_validation_ap': calibrated.pr_auc_ap})
    ranking = pd.DataFrame(rows)
    selected = ranking.loc[ranking.raw_validation_ap.idxmax()]
    assert selected.model == service.metadata['selected_model']
    assert selected.calibration == service.metadata['calibration']
    ranking.to_csv(ROOT/'reports/v2_selection_audit.csv', index=False)
    path = ROOT/'models/credit_risk.joblib'
    bundle = joblib.load(path)  # Already verified by RiskService.
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = bundle['metadata']
    metadata.update({'segments': segments, 'methodology_version': '2.0.0',
        'selection': 'V1 fitted model retained; audited raw validation AP selects Random Forest; validation Brier selects sigmoid; validation minimum 5*FN+FP selects review threshold',
        'low_segment_rule': 'Validation calibrated PD Q25/Q75, numpy linear interpolation; LOW < Q25, MEDIUM Q25 <= PD < Q75, HIGH >= Q75; independent of costs',
        'v1_artifact_sha256': original_hash})
    frozen = {key: metadata[key] for key in ['selected_model', 'calibration', 'threshold', 'segments', 'costs', 'selection', 'low_segment_rule', 'seed', 'methodology_version', 'v1_artifact_sha256']}
    (ROOT/'reports/frozen_decisions.json').write_text(json.dumps(frozen, indent=2))
    # Descriptive test summaries only, after boundaries were frozen.
    stored = pd.read_csv(ROOT/'reports/test_predictions.csv')
    test = data.loc[stored.ID]
    ptest = service.predictor.predict_proba(test[RAW_FEATURES])[:, 1]
    np.testing.assert_allclose(ptest, stored[metadata['selected_model']], atol=1e-12, rtol=1e-12)
    segment_rows = []
    for label, yy, pp in [('validation', validation[TARGET].to_numpy(), probabilities), ('test', test[TARGET].to_numpy(), ptest)]:
        frame = pd.DataFrame({'segment': [risk_category(p, segments) for p in pp], 'actual': yy, 'pd': pp})
        for segment, group in frame.groupby('segment'):
            segment_rows.append({'split': label, 'segment': segment, 'n': len(group),
                'observed_default_rate': group.actual.mean(), 'mean_predicted_pd': group.pd.mean()})
    pd.DataFrame(segment_rows).to_csv(ROOT/'reports/risk_segments.csv', index=False)
    fairness_audit(test, test[TARGET], ptest, metadata['threshold']).to_csv(ROOT/'reports/fairness_audit.csv', index=False)
    profiles = {}
    for segment in ['LOW', 'MEDIUM', 'HIGH']:
        indices = np.flatnonzero([risk_category(p, segments) == segment for p in ptest])
        profiles[segment] = {k: int(v) for k, v in test[RAW_FEATURES].iloc[indices[len(indices)//2]].items()}
    (ROOT/'app/demo_profiles.json').write_text(json.dumps(profiles, indent=2))
    joblib.dump(bundle, path, compress=3)
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    for target in ['models/metadata.json', 'reports/experiment_summary.json']:
        (ROOT/target).write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'segments': segments, 'model_retained': selected.model, 'prediction_rows_unchanged': len(ptest)}, indent=2))


if __name__ == '__main__':
    run()
