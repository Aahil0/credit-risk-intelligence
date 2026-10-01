import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                            recall_score, f1_score, confusion_matrix, brier_score_loss, log_loss)


def evaluate(y, probability, threshold=0.5, fn_cost=5, fp_cost=1):
    pred = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, probability)),
            'pr_auc_ap': float(average_precision_score(y, probability)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'brier': float(brier_score_loss(y, probability)),
            'log_loss': float(log_loss(y, probability)),
            'threshold': float(threshold), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'cost_per_customer': float((fn_cost*fn+fp_cost*fp)/len(y)), 'n': len(y)}


def threshold_table(y, p, fn_cost=5, fp_cost=1):
    return [evaluate(y, p, round(float(t), 2), fn_cost, fp_cost) for t in np.linspace(.01, .99, 99)]
