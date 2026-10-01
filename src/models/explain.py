"""Explain underlying score; calibration is a separate monotone mapping."""
import numpy as np
import shap
from src.features.names import feature_label
from src.features.engineering import engineer
from src.config import STATUS


def make_explainer(pipeline, background):
    model = pipeline.named_steps['model']
    transformed = pipeline.named_steps['prep'].transform(background)
    if model.__class__.__name__ == 'LogisticRegression':
        return shap.LinearExplainer(model, transformed), 'uncalibrated log-odds'
    unit = 'uncalibrated log-odds' if model.__class__.__name__ == 'XGBClassifier' else 'uncalibrated probability'
    return shap.TreeExplainer(model), unit


def explain(pipeline, explainer, X):
    values = explainer(pipeline.named_steps['prep'].transform(X))
    if values.values.ndim == 3:
        values = values[:, :, 1]
    names = pipeline.named_steps['prep'].get_feature_names_out().tolist()
    values.feature_names = names
    return values


def factors(explanation, top_n=8, original=None):
    v = explanation.values[0]
    data = explanation.data[0]
    readable = engineer(original).iloc[0] if original is not None else None
    def display_value(name, value):
        if readable is None:
            return {'feature_value': float(value), 'value_unit': 'transformed model input'}
        if name in readable.index:
            unit = 'NT$' if name == 'LIMIT_BAL' or name.startswith(('BILL_AMT', 'PAY_AMT')) else 'historical derived value (unscaled)'
            return {'feature_value': float(readable[name]), 'value_unit': unit}
        field, code = name.rsplit('_', 1)
        if field in STATUS:
            return {'feature_value': float(readable[field]), 'value_unit': 'repayment code',
                    'indicator_active': bool(value), 'indicator_code': int(float(code))}
        raise ValueError(f'No readable value mapping for {name}')
    return [{'feature': explanation.feature_names[i], 'label': feature_label(explanation.feature_names[i]), 'contribution': float(v[i]),
             'direction': 'increases_risk' if v[i] > 0 else 'decreases_risk',
             'transformed_feature_value': float(data[i]),
             'indicator_active': None, 'indicator_code': None,
             **display_value(explanation.feature_names[i], data[i])}
            for i in np.argsort(-np.abs(v))[:top_n] if abs(v[i]) > 1e-12]
