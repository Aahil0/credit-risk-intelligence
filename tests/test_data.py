"""Schema validation tests use synthetic fixtures, never claimed as model results."""
import numpy as np
import pandas as pd
import pytest
from src.config import RAW_FEATURES, DEMOGRAPHICS, TARGET
from src.data.acquire import validate


@pytest.fixture
def valid_frame():
    df = pd.DataFrame({c: np.zeros(30000, dtype=int) for c in RAW_FEATURES + DEMOGRAPHICS + [TARGET]})
    df['ID'] = np.arange(1,30001)
    df['LIMIT_BAL'] = 20000
    df['SEX'], df['EDUCATION'], df['MARRIAGE'], df['AGE'] = 2, 2, 1, 30
    return df


def test_source_schema_accepts_signed_bills_and_extra_demographic_codes(valid_frame):
    valid_frame.loc[0,'BILL_AMT1'] = -100
    valid_frame.loc[0,'EDUCATION'] = 6
    valid_frame.loc[0,'MARRIAGE'] = 0
    assert len(validate(valid_frame)) == 30000


@pytest.mark.parametrize('column,value', [('ID',2), (TARGET,2), ('LIMIT_BAL',0), ('PAY_AMT1',-1), ('PAY_0',10), ('SEX',3), ('AGE',0), ('BILL_AMT1',np.nan), ('BILL_AMT1',np.inf), ('PAY_AMT1',np.inf)])
def test_invalid_source_rejected(valid_frame,column,value):
    if isinstance(value, float):
        valid_frame[column] = valid_frame[column].astype(float)
    valid_frame.loc[0,column] = value
    with pytest.raises(ValueError):
        validate(valid_frame)


def test_missing_source_column_rejected(valid_frame):
    with pytest.raises(ValueError, match='schema'):
        validate(valid_frame.drop(columns='PAY_0'))
