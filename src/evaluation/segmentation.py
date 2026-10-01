"""Descriptive validation-score quartiles, independent of operating costs."""
import numpy as np


def define_segments(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Expected finite probabilities in [0, 1]')
    low, high = np.quantile(values, [.25, .75], method='linear')
    if low >= high:
        raise ValueError('Collapsed validation quartiles; risk segmentation needs review')
    return {'low_upper_exclusive': float(low), 'high_lower_inclusive': float(high)}


def review_flag(probability, threshold):
    return probability >= threshold
