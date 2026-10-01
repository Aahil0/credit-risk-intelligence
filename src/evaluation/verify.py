"""Recompute evidence from held-out rows and the shipped artifact."""
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from src.config import ROOT, RAW_FEATURES, TARGET
from src.models.service import RiskService
from src.evaluation.metrics import evaluate
from src.evaluation.segmentation import define_segments
from src.evaluation.fairness import fairness_audit
from src.data.acquire import validate


def run():
    service = RiskService()
    with zipfile.ZipFile(ROOT/'data/raw/uci.zip') as z:
        with z.open(next(n for n in z.namelist() if n.endswith('.xls'))) as f:
            df = pd.read_excel(f, header=1)
    validate(df)
    assert hashlib.sha256((ROOT/'data/raw/uci.zip').read_bytes()).hexdigest() == service.metadata['data_sha256']
    manifest = pd.read_csv(ROOT/'reports/split_manifest.csv', dtype={'group': str})
    predictions = pd.read_csv(ROOT/'reports/test_predictions.csv')
    test = df.set_index('ID').loc[predictions.ID]
    np.testing.assert_array_equal(test[TARGET], predictions.y_true)
    actual = service.predictor.predict_proba(test[RAW_FEATURES])[:,1]
    np.testing.assert_allclose(actual, predictions[service.metadata['selected_model']], atol=1e-12, rtol=1e-12)
    assert manifest.groupby('group')['split'].nunique().max() == 1
    ordered = df.set_index('ID').loc[manifest.ID]
    actual_groups = pd.util.hash_pandas_object(ordered[RAW_FEATURES], index=False).astype(str)
    np.testing.assert_array_equal(actual_groups.to_numpy(), manifest.group.to_numpy())
    validation = df.set_index('ID').loc[manifest.loc[manifest.split == 'validation', 'ID']]
    cuts = define_segments(service.predictor.predict_proba(validation[RAW_FEATURES])[:, 1])
    assert cuts == service.metadata['segments']
    metadata = json.loads((ROOT/'models/metadata.json').read_text())
    assert metadata == service.metadata
    frozen = json.loads((ROOT/'reports/frozen_decisions.json').read_text())
    assert all(metadata[key] == value for key, value in frozen.items())
    audit = fairness_audit(test, test[TARGET], actual, metadata['threshold'])
    stored_audit = pd.read_csv(ROOT/'reports/fairness_audit.csv', dtype={'group': str})
    pd.testing.assert_frame_equal(audit.reset_index(drop=True), stored_audit, check_dtype=False, atol=1e-12, rtol=1e-12)
    table = pd.read_csv(ROOT/'reports/test_model_comparison.csv')
    for _, row in table.iterrows():
        result = evaluate(predictions.y_true, predictions[row.model], row.threshold)
        for key, value in result.items():
            np.testing.assert_allclose(value, row[key], atol=1e-12)
    result = evaluate(test[TARGET], actual, service.metadata['threshold'])
    for key, value in result.items():
        np.testing.assert_allclose(value, service.metadata['test_metrics'][key], atol=1e-12)
    artifact = ROOT/'models/credit_risk.joblib'
    report = {'status': 'passed', 'held_out_rows_repredicted': len(test),
              'model_checksum': hashlib.sha256(artifact.read_bytes()).hexdigest(),
              'checks': ['Raw held-out labels match stored predictions', 'Artifact predictions match executed experiment',
                         'All comparator metrics recomputed from stored predictions', 'Metadata metrics match',
                         'No model-input group crosses split boundaries', 'Segment cuts recomputed from validation only',
                         'External, embedded and frozen metadata agree', 'Fairness rates and intervals recomputed'], 'metrics': result}
    (ROOT/'reports/artifact_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    run()
