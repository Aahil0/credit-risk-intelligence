"""Run a new experiment; existing V1 held-out evidence is historical, not fresh validation."""
import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from src.config import ROOT, SEED, RAW_FEATURES, DEMOGRAPHICS, TARGET, FN_COST, FP_COST
from src.data.acquire import load_data
from src.features.engineering import preprocessing, engineer
from src.evaluation.metrics import evaluate, threshold_table
from src.models.explain import make_explainer, explain
from src.evaluation.segmentation import define_segments
from src.evaluation.fairness import fairness_audit


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))


def savefig(name):
    plt.tight_layout()
    plt.savefig(ROOT / f'reports/figures/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for name in ['models', 'reports/figures', 'data/processed']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df[RAW_FEATURES], df[TARGET]
    # Group identical deployed inputs, including groups with conflicting labels.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    folds = np.zeros(len(df), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for k, (_, holdout) in enumerate(splitter.split(X, y, groups)):
        folds[holdout] = k
    masks = {'train': folds < 6, 'calibration': folds == 6, 'validation': folds == 7, 'test': folds >= 8}
    split_names = np.select(list(masks.values()), list(masks.keys()), default='unknown')
    pd.DataFrame({'ID': df.ID, 'split': split_names, 'group': groups}).to_csv(ROOT/'reports/split_manifest.csv', index=False)
    split_summary = {name: {'rows': int(mask.sum()), 'default_rate': float(y[mask].mean())} for name, mask in masks.items()}
    save_json(ROOT/'reports/splits.json', {'seed': SEED, 'method': '10-fold StratifiedGroupKFold; 6/1/1/2 folds',
             'groups': int(len(np.unique(groups))), 'duplicate_input_rows': int(len(df)-len(np.unique(groups))), 'partitions': split_summary})
    tr, ca, va, te = [masks[n] for n in ['train', 'calibration', 'validation', 'test']]
    logging.info('Partitions: %s', split_summary)
    # EDA on training only; target/count checks above are acquisition validation.
    df[tr].describe().to_csv(ROOT/'reports/training_descriptive_statistics.csv')
    y[tr].value_counts().sort_index().plot.bar(color=['#35a7b5','#ef7354'], title='Training class balance (0: no default, 1: default)')
    plt.ylabel('Customers'); savefig('class_balance')
    df[tr].groupby('PAY_0')[TARGET].agg(['mean', 'count']).to_csv(ROOT/'reports/delinquency_eda.csv')
    df[tr].groupby('PAY_0')[TARGET].mean().plot.bar(title='Training default rate by latest repayment code')
    plt.ylabel('Observed default rate'); savefig('delinquency_default_rate')
    engineered = engineer(X[tr])
    engineered[['utilization_recent', 'delinquency_frequency', 'payment_bill_ratio_mean']].hist(bins=35, figsize=(12, 4))
    savefig('engineered_distributions')
    corr = engineered.select_dtypes('number').corrwith(y[tr]).sort_values()
    corr.to_csv(ROOT/'reports/training_target_correlations.csv', header=['pearson_correlation'])

    specifications = {
        'Logistic Regression': (LogisticRegression(max_iter=2500, random_state=SEED),
                                {'model__C': [.01, .1, 1.0], 'model__class_weight': [None, 'balanced']}),
        'Random Forest': (RandomForestClassifier(n_estimators=160, random_state=SEED, n_jobs=2),
                          {'model__max_depth': [8, 14], 'model__min_samples_leaf': [10, 30], 'model__class_weight': [None]}),
        'XGBoost': (XGBClassifier(n_estimators=200, learning_rate=.05, tree_method='hist',
                                 subsample=.85, colsample_bytree=.85, random_state=SEED, n_jobs=2, eval_metric='logloss'),
                    {'model__max_depth': [2, 4], 'model__min_child_weight': [5, 15]})}
    # Identical input groups must stay together in every tuning fold too.
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_results, models, all_rows = [], {}, []
    for name, (estimator, grid) in specifications.items():
        logging.info('Tuning %s', name)
        pipe = Pipeline([('prep', preprocessing()), ('model', estimator)])
        search = GridSearchCV(pipe, grid, scoring={'ap': 'average_precision', 'roc': 'roc_auc'},
                              refit='ap', cv=cv, n_jobs=1, return_train_score=False, error_score='raise')
        search.fit(X[tr], y[tr], groups=groups[tr])
        raw = search.best_estimator_
        cv_table = pd.DataFrame(search.cv_results_)
        cv_table['model_name'] = name
        cv_results.append(cv_table)
        variants = {'none': raw}
        for method in ['sigmoid', 'isotonic']:
            variants[method] = CalibratedClassifierCV(FrozenEstimator(raw), method=method).fit(X[ca], y[ca])
        calibration_rows = []
        for method, candidate in variants.items():
            r = evaluate(y[va], candidate.predict_proba(X[va])[:, 1])
            calibration_rows.append({'model': name, 'calibration': method, **r})
        pd.DataFrame(calibration_rows).to_csv(ROOT/f'reports/calibration_{name.lower().replace(" ", "_")}.csv', index=False)
        # Choose calibrated output by lowest validation Brier. Brier also includes resolution.
        chosen = min(calibration_rows, key=lambda r: r['brier'])['calibration']
        calibrated = variants[chosen]
        pval = calibrated.predict_proba(X[va])[:, 1]
        row = {'model': name, 'calibration': chosen, 'cv_ap': float(search.best_score_),
               'raw_validation_ap': float(average_precision_score(y[va], raw.predict_proba(X[va])[:, 1])),
               'raw_validation_roc_auc': float(roc_auc_score(y[va], raw.predict_proba(X[va])[:, 1])),
               'best_params': search.best_params_, **evaluate(y[va], pval)}
        all_rows.append(row)
        models[name] = {'raw': raw, 'predictor': calibrated, 'calibration': chosen, 'validation': row}
        logging.info('%s validation AP %.4f / ROC %.4f; calibration=%s', name, row['pr_auc_ap'], row['roc_auc'], chosen)
    pd.concat(cv_results, ignore_index=True).to_csv(ROOT/'reports/cv_results.csv', index=False)
    pd.DataFrame(all_rows).to_csv(ROOT/'reports/validation_model_comparison.csv', index=False)
    pd.DataFrame(all_rows).to_csv(ROOT/'reports/v2_selection_audit.csv', index=False)
    # Freeze all choices BEFORE exposing test outcomes; never reselect on test.
    selected = max(all_rows, key=lambda r: r['raw_validation_ap'])['model']
    chosen = models[selected]
    pval = chosen['predictor'].predict_proba(X[va])[:, 1]
    thresholds = threshold_table(y[va], pval, FN_COST, FP_COST)
    pd.DataFrame(thresholds).to_csv(ROOT/'reports/validation_thresholds.csv', index=False)
    optimum = min(thresholds, key=lambda r: (r['cost_per_customer'], -r['threshold']))
    threshold = optimum['threshold']
    segments = define_segments(pval)
    low_cut, high_cut = segments['low_upper_exclusive'], segments['high_lower_inclusive']
    frozen = {'selected_model': selected, 'calibration': chosen['calibration'], 'threshold': threshold,
              'segments': segments, 'costs': {'false_negative': FN_COST, 'false_positive': FP_COST},
              'selection': 'Grouped CV AP tuning; raw validation AP for model family; validation Brier for calibration (log loss and reliability diagnostics retained); validation minimum 5*FN+FP for threshold',
              'low_segment_rule': 'Validation calibrated PD quartiles (linear interpolation): LOW below Q25, HIGH at/above Q75; independent of review threshold',
              'seed': SEED, 'methodology_version': '2.0.0'}
    save_json(ROOT/'reports/frozen_decisions.json', frozen)
    logging.info('Frozen decisions: %s', frozen)
    test_rows, predictions = [], pd.DataFrame({'ID': df.ID[te], 'y_true': y[te]})
    for name, item in models.items():
        p = item['predictor'].predict_proba(X[te])[:, 1]
        predictions[name] = p
        test_rows.append({'model': name, 'calibration': item['calibration'],
                          **evaluate(y[te], p, threshold if name == selected else .5)})
    predictions.to_csv(ROOT/'reports/test_predictions.csv', index=False)
    pd.DataFrame(test_rows).to_csv(ROOT/'reports/test_model_comparison.csv', index=False)
    ptest = predictions[selected].to_numpy()
    selected_metrics = evaluate(y[te], ptest, threshold)
    default_metrics = evaluate(y[te], ptest, .5)
    # Paired record bootstrap on frozen test predictions; no retraining or reselection.
    rng = np.random.default_rng(SEED)
    bootstrap = []
    yt = y[te].to_numpy()
    for _ in range(400):
        idx = rng.integers(0, len(yt), len(yt))
        bootstrap.append([roc_auc_score(yt[idx], ptest[idx]), average_precision_score(yt[idx], ptest[idx])])
    ci = np.quantile(bootstrap, [.025, .975], axis=0)
    save_json(ROOT/'reports/bootstrap_ci.json', {'method': '400 record-bootstrap samples; frozen model, IID approximation',
              'roc_auc_95': ci[:, 0].tolist(), 'pr_auc_ap_95': ci[:, 1].tolist()})
    # Separate sensitivity scenarios selected on validation, then evaluated on test.
    costs = []
    for ratio in [1, 2, 5, 10]:
        opt = min(threshold_table(y[va], pval, ratio, 1), key=lambda r: (r['cost_per_customer'], -r['threshold']))
        costs.append({'fn_fp_ratio': ratio, 'validation_cost': opt['cost_per_customer'],
                      **evaluate(y[te], ptest, opt['threshold'], ratio, 1)})
    pd.DataFrame(costs).to_csv(ROOT/'reports/cost_sensitivity.csv', index=False)
    def risk(p):
        return np.where(p < low_cut, 'LOW', np.where(p < high_cut, 'MEDIUM', 'HIGH'))
    segment_rows = []
    for label, yy, pp in [('validation', y[va].to_numpy(), pval), ('test', yt, ptest)]:
        temp = pd.DataFrame({'segment': risk(pp), 'actual': yy, 'pd': pp})
        for seg, group in temp.groupby('segment'):
            segment_rows.append({'split': label, 'segment': seg, 'n': len(group), 'observed_default_rate': group.actual.mean(), 'mean_predicted_pd': group.pd.mean()})
    pd.DataFrame(segment_rows).to_csv(ROOT/'reports/risk_segments.csv', index=False)
    demographic_test = df[te][DEMOGRAPHICS].copy()
    fairness_audit(demographic_test, yt, ptest, threshold).to_csv(ROOT/'reports/fairness_audit.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in models:
        RocCurveDisplay.from_predictions(yt, predictions[name], name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yt, predictions[name], name=name, ax=axes[1])
    axes[1].axhline(yt.mean(), linestyle='--', color='gray', label='Prevalence'); axes[1].legend(fontsize=8)
    savefig('model_curves')
    ConfusionMatrixDisplay.from_predictions(yt, ptest >= threshold, display_labels=['No default','Default'], cmap='Blues')
    plt.title(f'{selected}: frozen threshold {threshold:.2f}'); savefig('confusion_matrix')
    for label, model in [('Uncalibrated', chosen['raw']), ('Selected output', chosen['predictor'])]:
        frac, means = calibration_curve(yt, model.predict_proba(X[te])[:, 1], n_bins=10, strategy='quantile')
        plt.plot(means, frac, marker='o', label=label)
    plt.plot([0,1],[0,1],'--', color='gray'); plt.xlabel('Mean predicted PD'); plt.ylabel('Observed default fraction'); plt.legend(); savefig('calibration_curve')
    tdf = pd.DataFrame(thresholds)
    plt.plot(tdf.threshold, tdf.cost_per_customer, label='5 × FN + FP, per customer')
    plt.axvline(threshold, color='#ef7354', linestyle='--', label=f'Selected: {threshold:.2f}')
    plt.xlabel('Validation threshold'); plt.ylabel('Illustrative cost'); plt.legend(); savefig('threshold_cost')
    background = X[tr].sample(100, random_state=SEED)
    explainer, unit = make_explainer(chosen['raw'], background)
    sample = X[te].sample(min(500, te.sum()), random_state=SEED)
    explanations = explain(chosen['raw'], explainer, sample)
    shap.plots.beeswarm(explanations, max_display=15, show=False)
    savefig('shap_summary')
    importance = pd.Series(np.abs(explanations.values).mean(axis=0), index=explanations.feature_names).sort_values(ascending=False)
    importance.to_csv(ROOT/'reports/shap_importance.csv', header=['mean_absolute_shap'])
    importance.head(15).sort_values().plot.barh(figsize=(9,6), color='#35a7b5')
    plt.xlabel(f'Mean absolute SHAP ({unit})'); savefig('global_importance')
    # Demonstration cases from held-out data, without ID or demographics.
    profiles = {}
    for seg in ['LOW','MEDIUM','HIGH']:
        eligible = np.flatnonzero(risk(ptest) == seg)
        i = eligible[len(eligible)//2] if len(eligible) else 0
        profiles[seg] = {k: int(v) for k, v in X[te].iloc[i].items()}
    save_json(ROOT/'app/demo_profiles.json', profiles)
    example = pd.DataFrame([profiles['HIGH']])
    shap.plots.waterfall(explain(chosen['raw'], explainer, example)[0], max_display=12, show=False)
    savefig('individual_explanation')
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','shap','joblib','streamlit','fastapi','pydantic','matplotlib','scipy']}
    metadata = {**frozen, 'test_metrics': selected_metrics, 'test_metrics_at_0_5': default_metrics,
                'validation_metrics': optimum, 'explanation_unit': unit,
                'training_ranges': {c: [float(X[tr][c].min()), float(X[tr][c].max())] for c in RAW_FEATURES},
                'python': platform.python_version(), 'packages': versions,
                'trained_at_utc': datetime.now(timezone.utc).isoformat(),
                'data_sha256': hashlib.sha256((ROOT/'data/raw/uci.zip').read_bytes()).hexdigest()}
    joblib.dump({'raw': chosen['raw'], 'predictor': chosen['predictor'], 'background': background,
                 'metadata': metadata}, ROOT/'models/credit_risk.joblib', compress=3)
    save_json(ROOT/'models/metadata.json', metadata)
    digest = hashlib.sha256((ROOT/'models/credit_risk.joblib').read_bytes()).hexdigest()
    (ROOT/'models/credit_risk.sha256').write_text(digest+'\n')
    save_json(ROOT/'reports/experiment_summary.json', metadata)
    logging.info('Finished. Test metrics: %s', selected_metrics)

if __name__ == '__main__':
    run()
