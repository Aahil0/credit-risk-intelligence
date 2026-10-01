"""Offline release-download contracts; no unit test contacts GitHub."""
import hashlib
from http.client import IncompleteRead
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from src.config import ROOT
from src.models import artifact_download as download


class Response(BytesIO):
    status = 200


@pytest.fixture
def successful_download(monkeypatch):
    payload = b'synthetic model bytes for transport tests only'
    monkeypatch.setattr(download, 'EXPECTED_SHA256', hashlib.sha256(payload).hexdigest())
    calls = []
    def open_release():
        calls.append(True)
        return Response(payload)
    monkeypatch.setattr(download, '_open_release', open_release)
    return payload, calls


def test_release_contract_is_pinned():
    assert download.RELEASE_URL == 'https://github.com/Aahil0/credit-risk-intelligence/releases/download/v2.0.0/credit_risk.joblib'
    assert download.EXPECTED_SHA256 == '8426e02f00fb7d3dce1d9ea345986f1ec5093fda0bf83b8e01521ec5a1953180'


def test_existing_artifact_does_not_call_network(tmp_path, monkeypatch):
    target = tmp_path/'credit_risk.joblib'
    target.write_bytes(b'existing caller-managed artifact')
    monkeypatch.setattr(download, '_open_release', lambda: pytest.fail('Existing artifact must not download'))
    assert download.ensure_model_artifact(target) == target
    assert target.read_bytes() == b'existing caller-managed artifact'


def test_missing_default_artifact_installed_atomically_and_idempotently(tmp_path, monkeypatch, successful_download):
    payload, calls = successful_download
    monkeypatch.setattr(download, 'ROOT', tmp_path)
    target = tmp_path/'models/credit_risk.joblib'
    real_replace = download.os.replace
    installed = []
    def replace(source, destination):
        assert source.parent == destination.parent
        assert source.read_bytes() == payload
        assert not destination.exists()
        installed.append(True)
        real_replace(source, destination)
    monkeypatch.setattr(download.os, 'replace', replace)
    assert download.ensure_model_artifact() == target
    assert download.ensure_model_artifact() == target
    assert target.read_bytes() == payload
    assert calls == [True] and installed == [True]
    assert list(target.parent.iterdir()) == [target]


def test_checksum_mismatch_not_installed_and_temp_removed(tmp_path, monkeypatch):
    target = tmp_path/'models/credit_risk.joblib'
    monkeypatch.setattr(download, '_open_release', lambda: Response(b'untrusted bytes'))
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        download.ensure_model_artifact(target)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


@pytest.mark.parametrize('error', [URLError('offline'), HTTPError(download.RELEASE_URL, 404, 'Not Found', {}, None)])
def test_network_or_http_errors_are_clear_and_leave_no_temp(tmp_path, monkeypatch, error):
    target = tmp_path/'models/credit_risk.joblib'
    def fail():
        raise error
    monkeypatch.setattr(download, '_open_release', fail)
    with pytest.raises(RuntimeError, match='Cannot obtain the validated model release v2.0.0'):
        download.ensure_model_artifact(target)
    assert not target.exists() and list(target.parent.iterdir()) == []


def test_unsuccessful_http_response_not_installed(tmp_path, monkeypatch):
    response = Response(b'not an artifact')
    response.status = 503
    monkeypatch.setattr(download, '_open_release', lambda: response)
    with pytest.raises(RuntimeError, match='HTTP 503'):
        download.ensure_model_artifact(tmp_path/'model.joblib')
    assert list(tmp_path.iterdir()) == []


def test_partial_download_failure_removes_temp(tmp_path, monkeypatch):
    class BrokenResponse(Response):
        def read(self, size=-1):
            if self.tell():
                raise IncompleteRead(b'partial')
            return super().read(size)
    monkeypatch.setattr(download, '_open_release', lambda: BrokenResponse(b'partial download'))
    with pytest.raises(RuntimeError, match='Cannot obtain'):
        download.ensure_model_artifact(tmp_path/'model.joblib')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('operation', ['create', 'replace'])
def test_file_write_or_install_failure_is_clear_and_removes_temp(tmp_path, monkeypatch, successful_download, operation):
    def fail(*args, **kwargs):
        raise PermissionError('write denied')
    if operation == 'create':
        monkeypatch.setattr(download.tempfile, 'NamedTemporaryFile', fail)
    else:
        monkeypatch.setattr(download.os, 'replace', fail)
    with pytest.raises(RuntimeError, match='write denied'):
        download.ensure_model_artifact(tmp_path/'model.joblib')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('url', ['http://github.com/asset', 'https://example.com/asset',
    'https://github.com.evil.example/asset', 'https://user@github.com/asset',
    'https://github.com:8443/asset', 'https://github.com:invalid/asset'])
def test_untrusted_redirect_refused(url):
    with pytest.raises(RuntimeError, match='refused a redirect'):
        download._ReleaseRedirects().redirect_request(Request(download.RELEASE_URL), None, 302, '', {}, url)


def test_normal_github_asset_redirect_allowed():
    url = 'https://release-assets.githubusercontent.com/github-production-release-asset/asset'
    redirected = download._ReleaseRedirects().redirect_request(Request(download.RELEASE_URL), None, 302, '', {}, url)
    assert redirected.full_url == url


def test_opener_uses_fixed_url_and_timeout(monkeypatch):
    calls = []
    class Opener:
        def open(self, url, timeout):
            calls.append((url, timeout))
            return Response(b'')
    monkeypatch.setattr(download, 'build_opener', lambda handler: Opener())
    with download._open_release():
        pass
    assert calls == [(download.RELEASE_URL, 60)]


@pytest.mark.skipif(not (ROOT/'models/credit_risk.joblib').exists(), reason='Local validated artifact required')
def test_fresh_startup_mock_download_preserves_real_prediction(tmp_path, monkeypatch):
    from app.schemas import Applicant
    from src.models.service import RiskService
    local = ROOT/'models/credit_risk.joblib'
    payload = local.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == download.EXPECTED_SHA256
    applicant = Applicant(credit_limit=200000, repayment_status=[0]*6, bill_amounts=[45000]*6, payment_amounts=[5000]*6)
    baseline = RiskService(local).predict(applicant)
    monkeypatch.setattr(download, 'ROOT', tmp_path)
    (tmp_path/'models').mkdir()
    (tmp_path/'models/credit_risk.sha256').write_text(download.EXPECTED_SHA256+'\n')
    monkeypatch.setattr(download, '_open_release', lambda: Response(payload))
    fresh = RiskService()
    assert fresh.predict(applicant) == baseline
    assert (tmp_path/'models/credit_risk.joblib').read_bytes() == payload
    assert not list((tmp_path/'models').glob('*.tmp'))
