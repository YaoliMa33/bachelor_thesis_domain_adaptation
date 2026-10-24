"""Shared input fingerprints, experiment records and reproducible summaries."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_input(path, expected_sha256=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing input: {path}. Check --data and the dataset setup instructions.")
    data = path.read_bytes()
    digest = sha256(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Input checksum mismatch: {path}. Restore the published input; "
                         "use a separate --data path for intentionally modified data.")
    return data, digest


def runtime_provenance(data):
    """Record stable, path-independent metadata; no timestamps or output paths."""
    sources = [ROOT / 'experiment_io.py', ROOT / 'requirements.txt']
    for directory in ('st_training', 'uci_training'):
        sources.extend((ROOT / directory).rglob('*.py'))
    code = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes())
            for p in sorted(sources) if 'results' not in p.parts}
    commit, dirty = None, None
    if (ROOT / '.git').exists():
        try:
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=ROOT,
                                               stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
            commit = git('rev-parse', 'HEAD')
            dirty = bool(git('status', '--porcelain', '--untracked-files=normal'))
        except (OSError, subprocess.SubprocessError):
            pass
    return dict(data=data, git_commit=commit, git_dirty=dirty, code_sha256=code,
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 sklearn=sklearn.__version__, torch=str(torch.__version__),
                                 platform=platform.platform()))


def validate_run(config, seeds, methods):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 - 12000 for s in seeds)):
        raise ValueError('Seeds must be distinct integers from 0 through 4294955295.')
    if not methods or len(set(methods)) != len(methods) or set(methods) - {'DANN', 'DANNLC', 'CDAN', 'CDANLC'}:
        raise ValueError('Select distinct supported models in METHODS.')
    for key in ('epochs', 'steps_per_epoch', 'batch_size'):
        value = getattr(config, key)
        if type(value) is not int or value < (2 if key == 'batch_size' else 1):
            raise ValueError(f'{key} must be a positive integer (batch_size at least 2).')
    ramp = config.grl_ramp_updates or config.epochs * config.steps_per_epoch
    if type(ramp) is not int or ramp < 2:
        raise ValueError('The GRL ramp must contain at least two updates.')


def check_provenance(record, file, data, previous=None):
    provenance = record.get('provenance')
    required = {'data', 'git_commit', 'git_dirty', 'code_sha256', 'environment'}
    if (record.get('schema_version') != SCHEMA_VERSION or not isinstance(provenance, dict)
            or not required <= provenance.keys() or not isinstance(provenance['code_sha256'], dict)
            or not provenance['code_sha256'] or not isinstance(provenance['environment'], dict)
            or not {'python', 'numpy', 'sklearn', 'torch', 'platform'} <= provenance['environment'].keys()):
        raise ValueError(f'Missing or invalid provenance: {file}. Retrain in a new output directory.')
    if (any(not isinstance(v, str) or re.fullmatch(r'[0-9a-f]{64}', v) is None
            for v in provenance['code_sha256'].values())
            or any(not isinstance(v, str) or not v for v in provenance['environment'].values())
            or (provenance['git_commit'] is not None and
                (not isinstance(provenance['git_commit'], str) or
                 re.fullmatch(r'[0-9a-f]{40,64}', provenance['git_commit']) is None))
            or (provenance['git_dirty'] is not None and type(provenance['git_dirty']) is not bool)):
        raise ValueError(f'Invalid provenance fields: {file}. Retrain in a new output directory.')
    if provenance['data'] != data:
        raise ValueError(f'Data fingerprint mismatch: {file}. Use the original --data or retrain in a new output directory.')
    if previous is not None and provenance != previous:
        raise ValueError(f'Mixed training provenance: {file}. Keep data, code revision and training environment consistent.')
    return provenance


def load_result(file, *, dataset, method, seed, task, config, data, entropy_conditioning=False):
    file = Path(file)
    if not file.is_file():
        raise ValueError(f'Missing training record: {file}. Run training with matching output, methods, seeds and protocol.')
    try:
        record = json.loads(file.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid training record: {file}. Retrain this task.') from exc
    if not isinstance(record, dict):
        raise ValueError(f'Invalid training record object: {file}')
    normalize = lambda value: json.loads(json.dumps(value))
    expected = dict(dataset=dataset, method=method, seed=seed, config=normalize(config))
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    stored_task = record.get('task')
    # B1->B2 can serve either or both UCI protocols; memberships do not change training.
    if not isinstance(stored_task, dict) or any(stored_task.get(key) != normalize(task[key])
            for key in ('name', 'source', 'targets', 'evaluations')):
        mismatches.append('task')
    if (record.get('entropy_conditioning', False) is not entropy_conditioning
            or record.get('variant', method) != method + ('+E' if entropy_conditioning else '')):
        mismatches.append('variant (+E/base)')
    if mismatches:
        raise ValueError(f'Incompatible training record: {file}; mismatched ' + ', '.join(mismatches)
                         + '. Use matching settings or retrain in a new output directory.')
    check_provenance(record, file, data)
    return record


def validate_scores(record, batches, evaluation_batches, file):
    scores = record.get('scores')
    if not isinstance(scores, dict) or set(scores) != {str(b) for b in evaluation_batches}:
        raise ValueError(f'Invalid score batches: {file}')
    for batch in evaluation_batches:
        score = scores[str(batch)]
        labels = batches[batch][1]
        if (not isinstance(score, dict) or type(score.get('n')) is not int or score['n'] != len(labels)
                or not isinstance(score.get('predictions'), list) or len(score['predictions']) != len(labels)
                or any(type(p) is not int or not 0 <= p < record['config']['classes'] for p in score['predictions'])):
            raise ValueError(f'Invalid sample counts or predictions: {file}, batch {batch}')
        correct = int(np.count_nonzero(np.asarray(score['predictions']) == labels))
        accuracy = score.get('accuracy')
        if (type(score.get('correct')) is not int or score['correct'] != correct
                or type(accuracy) not in (int, float) or not math.isfinite(accuracy)
                or not math.isclose(accuracy, 100 * correct / len(labels), rel_tol=0, abs_tol=1e-10)):
            raise ValueError(f'Inconsistent accuracy or correct count: {file}, batch {batch}')


def validate_checkpoint(file, record):
    file = Path(file)
    if not file.is_file():
        raise ValueError(f'Missing checkpoint: {file}. Retrain this task.')
    content = file.read_bytes()
    if sha256(content) != record.get('checkpoint_sha256'):
        raise ValueError(f'Checkpoint hash mismatch: {file}. The model and JSON are not a matching pair; retrain in a new output directory.')
    return content


def check_existing(folder, method, record):
    file, checkpoint = folder / f'{method}.json', folder / f'{method}.pt'
    if not file.exists():
        if checkpoint.exists():
            raise ValueError(f'Checkpoint without its JSON record: {checkpoint}. Use a new output directory.')
        return
    old = load_result(file, dataset=record['dataset'], method=method, seed=record['seed'],
                      task=record['task'], config=record['config'], data=record['provenance']['data'],
                      entropy_conditioning=record.get('entropy_conditioning', False))
    check_provenance(old, file, record['provenance']['data'], record['provenance'])
    validate_checkpoint(checkpoint, old)


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def save_result(folder, method, model, record):
    folder = Path(folder)
    check_existing(folder, method, record)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Non-finite model parameters; no result was saved.')
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    content = buffer.getvalue()
    record = dict(record, checkpoint_sha256=sha256(content))
    encoded = json.dumps(record, indent=2, allow_nan=False).encode('utf-8')
    # JSON is written last. An interrupted pair is detected by its checkpoint hash.
    atomic_write(folder / f'{method}.pt', content)
    atomic_write(folder / f'{method}.json', encoded)


def summarize_accuracies(raw):
    accuracy = {m: dict(mean=statistics.mean(v), sd=statistics.stdev(v) if len(v) > 1 else None)
                for m, v in raw.items()}
    paired = {}
    for family in ('DANN', 'CDAN'):
        if family in raw and family + 'LC' in raw:
            differences = [b - a for a, b in zip(raw[family], raw[family + 'LC'])]
            paired[family] = dict(mean=statistics.mean(differences),
                                 sd=statistics.stdev(differences) if len(differences) > 1 else None,
                                 positive_seeds=sum(d > 0 for d in differences))
    return dict(accuracy=accuracy, paired_LC_gains=paired, seed_means=raw)
