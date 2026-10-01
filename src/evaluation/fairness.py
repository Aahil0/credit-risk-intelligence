"""Descriptive rates with Wilson intervals; no fairness certification.

Intervals condition on the fitted model and approximate independent records.
They exclude selection uncertainty, clustering and multiple-comparison adjustment.
"""
import numpy as np
import pandas as pd


def wilson(successes, total):
    if total < 30:
        return (None, None)
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    margin = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0., center-margin), min(1., center+margin)


def fairness_audit(demographics, y, probabilities, threshold):
    data = demographics.copy()
    data['AGE_BAND'] = pd.cut(data.AGE, [17, 29, 44, 59, 100], labels=['18-29', '30-44', '45-59', '60+'])
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    rows = []
    for attribute in ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE_BAND']:
        for group in data[attribute].dropna().unique():
            mask = (data[attribute] == group).to_numpy()
            yy, pp = y[mask], probabilities[mask]
            flags = pp >= threshold
            n, positives, negatives = len(yy), int(yy.sum()), int((yy == 0).sum())
            row = {'attribute': attribute, 'group': str(group), 'n': n,
                   'positive_n': positives, 'negative_n': negatives, 'small_group': n < 100,
                   'brier': float(np.mean((yy-pp)**2)), 'mean_predicted_pd': float(pp.mean()),
                   'calibration_gap': float(pp.mean()-yy.mean())}
            for metric, successes, total in [('default_rate', positives, n), ('flag_rate', int(flags.sum()), n),
                    ('tpr', int(flags[yy == 1].sum()), positives), ('fpr', int(flags[yy == 0].sum()), negatives)]:
                row[metric] = successes/total if total and (metric in ['default_rate', 'flag_rate'] or total >= 30) else None
                # Major groups only; small-group rates remain explicitly descriptive.
                lower, upper = wilson(successes, total) if n >= 100 else (None, None)
                row[f'{metric}_ci95_lower'], row[f'{metric}_ci95_upper'] = lower, upper
            rows.append(row)
    return pd.DataFrame(rows)
