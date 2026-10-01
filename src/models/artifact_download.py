"""Retrieve only the pinned project release; verify before installing joblib.

SHA-256 checks integrity against the reviewed bytes, not signatures or publisher
authenticity. Joblib deserialization executes code: trust the project and binary.
"""
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from src.config import ROOT

RELEASE_URL = 'https://github.com/Aahil0/credit-risk-intelligence/releases/download/v2.0.0/credit_risk.joblib'
EXPECTED_SHA256 = '8426e02f00fb7d3dce1d9ea345986f1ec5093fda0bf83b8e01521ec5a1953180'
_RELEASE_HOSTS = {'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}


class _ReleaseRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            parsed = urlsplit(newurl)
            port = parsed.port
        except ValueError as exc:
            raise RuntimeError('Model download refused a redirect with an invalid URL') from exc
        if (parsed.scheme != 'https' or parsed.hostname not in _RELEASE_HOSTS
                or parsed.username is not None or parsed.password is not None
                or port not in (None, 443)):
            raise RuntimeError('Model download refused a redirect outside trusted HTTPS GitHub release hosts')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_release():
    return build_opener(_ReleaseRedirects()).open(RELEASE_URL, timeout=60)


def ensure_model_artifact(model_path=None):
    """Return an existing artifact or atomically install verified release bytes.

The destination may be overridden for offline tests; the URL/hash cannot be
supplied by an applicant. Existing artifacts retain the service's normal check.
"""
    target = Path(model_path) if model_path is not None else ROOT/'models/credit_risk.joblib'
    if target.exists():
        return target
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _open_release() as response:
            if response.status != 200:
                raise RuntimeError(f'Model release download returned HTTP {response.status}')
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.credit_risk-',
                                             suffix='.tmp', delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024*1024):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != EXPECTED_SHA256:
                raise RuntimeError('Downloaded model SHA-256 checksum mismatch; artifact was not installed')
        os.replace(temporary, target)
        return target
    except (OSError, RuntimeError, HTTPException) as exc:
        raise RuntimeError('Cannot obtain the validated model release v2.0.0: '
                           f'{exc}. Check that credit_risk.joblib is published at the pinned release URL.') from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError('Cannot remove the partial model download; check models directory permissions') from exc
