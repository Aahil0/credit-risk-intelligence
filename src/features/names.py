"""Human-readable feature labels for explanations."""
MONTHS = ['September', 'August', 'July', 'June', 'May', 'April']
LABELS = {
    'LIMIT_BAL': 'Credit limit', 'utilization_recent': 'Latest credit utilization',
    'utilization_mean': 'Average credit utilization', 'utilization_max': 'Peak credit utilization',
    'over_limit_months': 'Months above credit limit', 'nonpositive_bill_months': 'Months with zero or credit statement',
    'payment_bill_ratio_recent': 'Latest payment / bill ratio', 'payment_bill_ratio_mean': 'Average payment / bill ratio',
    'zero_payment_months': 'Months without payment', 'payment_limit_mean': 'Average payment / credit limit',
    'delinquency_frequency': 'Months with payment delay', 'severe_delinquency_months': 'Months delayed by at least two months',
    'max_delay': 'Maximum payment delay', 'recent_delay_change': 'Change in recent payment delay',
    'bill_trend_per_limit': 'Bill trend relative to credit limit', 'payment_trend_per_limit': 'Payment trend relative to credit limit'
}


def feature_label(feature):
    if feature in LABELS:
        return LABELS[feature]
    if feature.startswith('PAY_') and not feature.startswith('PAY_AMT'):
        field, code = feature.rsplit('_', 1)
        fields = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
        if field in fields:
            return f'{MONTHS[fields.index(field)]} repayment code = {int(float(code))}'
    for prefix, label in [('BILL_AMT','bill amount'), ('PAY_AMT','payment amount')]:
        if feature.startswith(prefix):
            return f'{MONTHS[int(feature[len(prefix):])-1]} {label}'
    return feature.replace('_',' ').capitalize()
