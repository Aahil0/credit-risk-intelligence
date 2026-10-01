import pytest
from fastapi.testclient import TestClient
from app.api import app
from app.schemas import Applicant
from src.models.service import get_service
from src.config import ROOT

PAYLOAD = {'credit_limit': 200000, 'repayment_status': [0]*6,
           'bill_amounts': [45000,43000,40000,38000,35000,33000], 'payment_amounts': [5000]*6}


class FakeService:
    metadata = {'selected_model': 'Test double'}
    def predict(self, applicant):
        return {'probability_of_default': .2, 'risk_category': 'HIGH', 'flagged_for_review': True,
                'important_risk_factors': [], 'explanation_unit': 'test', 'explanation_base_value': .1,
                'explanation_total_contribution': .1, 'model': 'Test double', 'calibration': 'none',
                'decision_threshold': .15, 'warnings': [], 'notice': 'test only'}


@pytest.fixture
def contract_client(monkeypatch):
    # Contract tests run in CI even when trained artifacts are gitignored.
    monkeypatch.setattr('app.api.get_service', lambda: FakeService())
    with TestClient(app) as client:
        yield client


def test_api_contract(contract_client):
    assert contract_client.get('/health').status_code == 200
    response = contract_client.post('/predict', json=PAYLOAD)
    assert response.status_code == 200
    assert set(['probability_of_default','risk_category','important_risk_factors']) <= response.json().keys()


@pytest.mark.parametrize('change', [
    {'credit_limit': 0}, {'credit_limit': '10000'}, {'repayment_status': [0]*5},
    {'repayment_status': [10]*6}, {'payment_amounts': [-1]*6}, {'sex': 1},
    {'bill_amounts': [1.5]*6}])
def test_invalid_requests_rejected(contract_client, change):
    assert contract_client.post('/predict', json={**PAYLOAD, **change}).status_code == 422


def test_missing_fields_rejected(contract_client):
    assert contract_client.post('/predict', json={}).status_code == 422


@pytest.mark.skipif(not (ROOT/'models/credit_risk.joblib').exists(), reason='Run training for artifact integration')
def test_real_api_matches_shared_service_and_repeatable():
    with TestClient(app) as client:
        assert client.get('/health').json()['status'] == 'ok'
        first = client.post('/predict', json=PAYLOAD)
        assert first.status_code == 200
        assert first.json() == client.post('/predict', json=PAYLOAD).json()
        assert first.json() == get_service().predict(Applicant(**PAYLOAD))
        assert 0 <= first.json()['probability_of_default'] <= 1
        assert first.json()['important_risk_factors']
