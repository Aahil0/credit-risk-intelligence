"""Real local HTTP smoke tests; always terminate only processes started here."""
import json
import os
import socket
import subprocess
import sys
import time
import requests
from src.config import ROOT


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def run():
    results = {}
    payload = json.loads((ROOT/'reports/api_example_request.json').read_text())
    for name, module, arguments, health in [
        ('api', 'uvicorn', ['app.api:app', '--host', '127.0.0.1'], '/health'),
        ('streamlit', 'streamlit', ['run', 'app/dashboard.py', '--server.address=127.0.0.1', '--server.headless=true', '--browser.gatherUsageStats=false'], '/_stcore/health')]:
        port = free_port()
        arguments += ['--port' if name == 'api' else '--server.port', str(port)]
        env = dict(os.environ, MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
        with (ROOT/f'reports/v2_{name}_live.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', module, *arguments], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic()+45
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'{name} startup failed; inspect log')
                    try:
                        response = requests.get(base+health, timeout=1)
                        if response.status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'{name} health timeout')
                    time.sleep(.25)
                checks = ['HTTP health 200']
                if name == 'api':
                    response = requests.post(base+'/predict', json=payload, timeout=10)
                    response.raise_for_status()
                    prediction = response.json()
                    assert prediction == requests.post(base+'/predict', json=payload, timeout=10).json()
                    assert requests.post(base+'/predict', json={**payload, 'credit_limit': False}, timeout=10).status_code == 422
                    assert requests.get(base+'/openapi.json', timeout=5).status_code == 200
                    (ROOT/'reports/api_example_response.json').write_text(json.dumps(prediction, indent=2))
                    checks += ['HTTP predict 200', 'Repeatable response', 'Invalid input 422', 'OpenAPI 200']
                else:
                    assert requests.get(base, timeout=5).status_code == 200
                    checks += ['HTTP app shell 200; interactive rendering verified separately by AppTest']
                results[name] = {'status': 'passed', 'checks': checks}
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    (ROOT/'reports/v2_http_smoke.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    run()
