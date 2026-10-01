"""Shared inference path for API and dashboard."""
import hashlib
from functools import lru_cache
from pathlib import Path
import logging
import joblib
import pandas as pd
from src.config import ROOT, RAW_FEATURES, STATUS, BILLS, PAYMENTS
from src.models.explain import make_explainer, explain, factors
from src.evaluation.segmentation import review_flag


def risk_category(probability, segments):
    if probability < segments['low_upper_exclusive']:
        return 'LOW'
    if probability < segments['high_lower_inclusive']:
        return 'MEDIUM'
    return 'HIGH'


class RiskService:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path is not None else ROOT/'models/credit_risk.joblib'
        checksum_path = path.with_suffix('.sha256')
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum_path.read_text().strip():
            raise ValueError('Model checksum mismatch')
        # Joblib is executable serialization. Load ONLY trusted project artifacts.
        bundle = joblib.load(path)
        self.raw, self.predictor = bundle['raw'], bundle['predictor']
        self.metadata = bundle['metadata']
        self.explainer, self.unit = make_explainer(self.raw, bundle['background'])
        logging.info('Loaded %s (%s calibration)', self.metadata['selected_model'], self.metadata['calibration'])

    def predict(self, applicant):
        record = {'LIMIT_BAL': applicant.credit_limit}
        record.update(dict(zip(STATUS, applicant.repayment_status)))
        record.update(dict(zip(BILLS, applicant.bill_amounts)))
        record.update(dict(zip(PAYMENTS, applicant.payment_amounts)))
        X = pd.DataFrame([record], columns=RAW_FEATURES)
        p = float(self.predictor.predict_proba(X)[0, 1])
        explanation = explain(self.raw, self.explainer, X)
        warnings = []
        for name, value in record.items():
            lower, upper = self.metadata['training_ranges'][name]
            if not lower <= value <= upper:
                warnings.append(f'{name} is outside the observed training range [{lower:g}, {upper:g}].')
        all_factors = factors(explanation, top_n=len(explanation.feature_names), original=X)
        positive = [f for f in all_factors if f['contribution'] > 0][:4]
        negative = [f for f in all_factors if f['contribution'] < 0][:4]
        return {'probability_of_default': p,
                'risk_category': risk_category(p, self.metadata['segments']),
                'flagged_for_review': review_flag(p, self.metadata['threshold']),
                'important_risk_factors': sorted(positive + negative, key=lambda f: -abs(f['contribution'])),
                'explanation_unit': self.unit,
                'explanation_base_value': float(explanation.base_values[0]),
                'explanation_total_contribution': float(explanation.values[0].sum()),
                'model': self.metadata['selected_model'], 'calibration': self.metadata['calibration'],
                'decision_threshold': self.metadata['threshold'], 'warnings': warnings,
                'notice': 'Educational decision support only. Contributions explain the underlying uncalibrated score, not the calibrated PD, and are not causal.'}


@lru_cache(maxsize=1)
def get_service():
    return RiskService()
