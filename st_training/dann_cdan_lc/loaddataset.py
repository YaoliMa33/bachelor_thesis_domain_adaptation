"""Load supplied feature files and define the evaluation protocols."""
import csv
from dataclasses import dataclass
from pathlib import Path
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_io import read_input

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Task:
    name: str
    source: int
    targets: tuple
    evaluations: tuple
    memberships: tuple


def load_data(path):
    return load_data_with_fingerprint(path)[0]


def load_data_with_fingerprint(path):
    path = Path(path)
    default = ROOT / 'st_dataset/st_54d/features_54d.csv'
    expected = None
    if path.resolve() == default.resolve():
        expected = json.loads((ROOT / 'st_dataset/checksums.json').read_text(encoding='utf-8'))['files']['st_54d/features_54d.csv']
    content, digest = read_input(path, expected)
    descriptors = ('exp_mean', 'exp_absmax', 'exp_span', 'exp_tail', 'exp_auc_abs',
                   'rec_mean', 'rec_tail', 'rec_slope', 'overall_std')
    columns = [f's{i}_{feature}' for i in range(1, 7) for feature in descriptors]
    try:
        reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline=''))
        header = reader.fieldnames or []
        if (len(header) != len(set(header)) or not {'sample_id', 'label', 'batch'} <= set(header)
                or [c for c in header if any(c.startswith(f's{i}_') for i in range(1, 7))] != columns):
            raise ValueError('Expected unique headers and all 54 sensor features in published order.')
        rows = list(reader)
        if not rows:
            raise ValueError('No samples found.')
        ids = set()
        names = ['source', 'batch1', 'batch2', 'batch3', 'batch4', 'batch5']
        for line, row in enumerate(rows, 2):
            if (None in row or any(v is None or not v.strip() for v in row.values())
                    or row['sample_id'] in ids or row['batch'] not in names
                    or row['label'] not in ('air', 'alcohol', 'acetone')):
                raise ValueError(f'Invalid row, label, batch or duplicate sample_id at line {line}.')
            ids.add(row['sample_id'])
        batches = {}
        for batch, name in enumerate(names, 1):
            subset = [r for r in rows if r['batch'] == name]
            if not subset:
                raise ValueError(f'Missing batch: {name}')
            x = np.array([[float(r[c]) for c in columns] for r in subset], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f'Non-finite feature in batch {name}')
            # Class membership is determined only by label, never by source_kind.
            y = np.array([('air', 'alcohol', 'acetone').index(r['label']) for r in subset], dtype=np.int64)
            batches[batch] = x, y
    except (ValueError, UnicodeError, csv.Error) as exc:
        raise ValueError(f'Invalid ST data {path}: {exc}') from exc
    return batches, dict(dataset='ST', files={'features_54d.csv': digest})


def preprocess(source, arrays):
    transformed_source = np.arcsinh(source.astype(np.float64))
    scaler = StandardScaler().fit(transformed_source)
    return [torch.from_numpy(((np.arcsinh(x.astype(np.float64)) - scaler.mean_)
                              / scaler.scale_).astype(np.float32)) for x in arrays]


def tasks():
    result = [Task("S1_pool", 1, tuple(range(2, 7)), tuple(range(2, 7)),
                   tuple(("Setting1", b) for b in range(2, 7)))]
    for batch in range(2, 7):
        memberships = (("Setting2", batch),)
        if batch == 2:
            memberships += (("Setting3", batch),)
        result.append(Task(f"S2_B{batch}", 1, (batch,), (batch,), memberships))
        if batch > 2:
            result.append(Task(f"S3_to_B{batch}", 1, tuple(range(2, batch + 1)),
                               (batch,), (("Setting3", batch),)))
    return result
