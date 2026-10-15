"""Load supplied feature files and define the evaluation protocols."""
import csv
from dataclasses import dataclass
from pathlib import Path

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
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [c for c in reader.fieldnames
                   if any(c.startswith(f"s{i}_") for i in range(1, 7))]
        rows = list(reader)
    batches = {}
    for batch in range(1, 7):
        name = "source" if batch == 1 else f"batch{batch - 1}"
        subset = [r for r in rows if r["batch"] == name]
        x = np.array([[float(r[c]) for c in columns] for r in subset], dtype=np.float64)
        y = np.array([("air", "alcohol", "acetone").index(r["label"])
                      for r in subset], dtype=np.int64)
        batches[batch] = x, y
    return batches


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
