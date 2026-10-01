"""Execute review notebook in-process; no network/kernel sockets required.

Run from the repository root: python notebooks/run_walkthrough.py
All outputs are captured from actual IPython execution. This runner does not
simulate a Jupyter kernel or verify notebook frontend rendering.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root = Path(__file__).resolve().parents[1]
os.chdir(root)
path = root/'notebooks/01_experiment_walkthrough.ipynb'
nb = nbformat.read(path, as_version=4)
shell = InteractiveShell.instance()
count = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    count += 1
    with capture_output(stdout=True, stderr=True, display=True) as output:
        result = shell.run_cell(cell.source, store_history=True)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
    cell.execution_count = count
    cell.outputs = []
    for name, content in [('stdout', output.stdout), ('stderr', output.stderr)]:
        if content:
            cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=content))
    for rich in output.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
nb.metadata['execution_method'] = 'Actual in-process IPython execution; Jupyter socket transport unavailable in build environment'
nbformat.validate(nb)
nbformat.write(nb, path)
print(f'Executed {count} code cells; captured actual output in {path.name}')
