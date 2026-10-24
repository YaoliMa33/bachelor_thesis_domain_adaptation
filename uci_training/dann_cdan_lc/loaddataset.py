"""Load supplied feature files and define the evaluation protocols."""
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


def load_data(directory):
    return load_data_with_fingerprint(directory)[0]


def load_data_with_fingerprint(directory):
    directory = Path(directory)
    missing = [f'batch{b}.dat' for b in range(1, 11) if not (directory / f'batch{b}.dat').is_file()]
    if missing:
        raise ValueError("Missing UCI data. Run 'python uci_dataset_128d/download.py' or pass --data PATH. Missing: " + ', '.join(missing))
    expected = {}
    if directory.resolve() == (ROOT / 'uci_dataset_128d/data').resolve():
        expected = json.loads((ROOT / 'uci_dataset_128d/checksums.json').read_text(encoding='utf-8'))['files']
    batches, fingerprints = {}, {}
    for batch in range(1, 11):
        file = directory / f'batch{batch}.dat'
        content, fingerprints[file.name] = read_input(file, expected.get(file.name))
        features, labels = [], []
        line_number = 0
        try:
            for line_number, line in enumerate(content.decode('utf-8').splitlines(), 1):
                label, values = line.split(';', 1)
                label = int(label)
                tokens = values.split()
                if label not in range(1, 7) or len(tokens) != 129 or not np.isfinite(float(tokens[0])):
                    raise ValueError('Expected label 1-6, concentration and exactly 128 indexed features.')
                x = np.zeros(128, dtype=np.float32)
                seen = set()
                for entry in tokens[1:]:
                    index, value = entry.split(':', 1)
                    index, value = int(index), float(value)
                    if index not in range(1, 129) or index in seen or not np.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
                        raise ValueError('Invalid, duplicate or non-finite feature.')
                    seen.add(index)
                    x[index - 1] = value
                features.append(x)
                labels.append(label - 1)
            if not features:
                raise ValueError('Empty batch.')
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f'Invalid UCI data {file}, line {line_number}: {exc}') from exc
        batches[batch] = np.array(features), np.array(labels, dtype=np.int64)
    return batches, dict(dataset='UCI', files=fingerprints)


def preprocess(source, arrays):
    transformed_source = np.arcsinh(source.astype(np.float64))
    scaler = StandardScaler().fit(transformed_source)
    return [torch.from_numpy(((np.arcsinh(x.astype(np.float64)) - scaler.mean_)
                              / scaler.scale_).astype(np.float32)) for x in arrays]


def tasks(uci_protocol="both"):
    result = []
    if uci_protocol in ("long", "both"):
        for batch in range(2, 11):
            memberships = (("Long", batch),)
            if uci_protocol == "both" and batch == 2:
                memberships += (("Medium", batch),)
            result.append(Task(f"B1_B{batch}", 1, (batch,), (batch,), memberships))
    if uci_protocol in ("medium", "both"):
        first_source = 2 if uci_protocol == "both" else 1
        for source in range(first_source, 10):
            target = source + 1
            result.append(Task(f"B{source}_B{target}", source, (target,),
                               (target,), (("Medium", target),)))
    return result
