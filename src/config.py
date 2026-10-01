from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILLS = [f'BILL_AMT{i}' for i in range(1, 7)]
PAYMENTS = [f'PAY_AMT{i}' for i in range(1, 7)]
RAW_FEATURES = ['LIMIT_BAL'] + STATUS + BILLS + PAYMENTS
DEMOGRAPHICS = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
TARGET = 'default payment next month'
DATA_URL = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
FN_COST, FP_COST = 5, 1
