"""Download only from UCI; validate actual data and record provenance."""
import hashlib
import json
import logging
import zipfile
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
from src.config import ROOT, DATA_URL, RAW_FEATURES, DEMOGRAPHICS, TARGET, STATUS, PAYMENTS


def validate(df):
    required = ['ID'] + RAW_FEATURES + DEMOGRAPHICS + [TARGET]
    if set(df.columns) != set(required):
        raise ValueError('Unexpected dataset schema')
    if len(df) != 30000 or df.ID.nunique() != len(df):
        raise ValueError('Expected 30,000 unique client IDs')
    if df.isna().any().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError('Missing values or invalid target')
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError('Source values must be finite')
    if (df.LIMIT_BAL <= 0).any() or (df[PAYMENTS] < 0).any().any():
        raise ValueError('Invalid credit limits or payments')
    if not df[STATUS].isin(range(-2, 10)).all().all():
        raise ValueError('Unexpected repayment status')
    if not df.SEX.isin([1, 2]).all() or not df.EDUCATION.isin(range(7)).all() or not df.MARRIAGE.isin(range(4)).all() or not df.AGE.between(18, 100).all():
        raise ValueError('Unexpected demographics')
    return df


def load_data():
    path = ROOT / 'data/raw/uci.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(DATA_URL, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('.xls'))
        with archive.open(name) as f:
            df = pd.read_excel(f, header=1, engine='xlrd')
    validate(df)
    metadata = {'source': DATA_URL, 'dataset_doi': '10.24432/C55S3H',
                'license': 'CC BY 4.0', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'validated_at_utc': datetime.now(timezone.utc).isoformat(),
                'rows': len(df), 'missing_cells': int(df.isna().sum().sum()),
                'default_count': int(df[TARGET].sum()),
                'negative_bill_cells': int((df.filter(like='BILL_AMT') < 0).sum().sum()),
                'education_counts': df.EDUCATION.value_counts().sort_index().to_dict(),
                'marriage_counts': df.MARRIAGE.value_counts().sort_index().to_dict()}
    (ROOT / 'reports/data_validation.json').write_text(json.dumps(metadata, indent=2))
    logging.info('Validated %s rows from UCI', len(df))
    return df

if __name__ == '__main__':
    load_data()
