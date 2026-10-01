import pytest
from streamlit.testing.v1 import AppTest
from src.config import ROOT


@pytest.mark.skipif(not (ROOT/'models/credit_risk.joblib').exists(), reason='Train model before dashboard integration')
def test_dashboard_all_profiles_render_predictions():
    at = AppTest.from_file(str(ROOT/'app/dashboard.py'), default_timeout=40).run()
    assert not at.exception
    for profile in ['LOW','MEDIUM','HIGH']:
        at.selectbox[0].select(profile).run()
        at.button[0].click().run()
        assert not at.exception
        assert at.metric[0].label == 'Probability of default'
        assert at.metric[1].value == profile
