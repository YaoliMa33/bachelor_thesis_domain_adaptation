"""Load supplied feature files and define the evaluation protocols."""
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


def load_data(directory):
    batches = {}
    for batch in range(1, 11):
        features, labels = [], []
        for line in (Path(directory) / f"batch{batch}.dat").read_text().splitlines():
            label, values = line.split(";", 1)
            x = np.zeros(128, dtype=np.float32)
            # The first token after ';' is concentration, not a sensor feature.
            for entry in values.split()[1:]:
                index, value = entry.split(":")
                x[int(index) - 1] = float(value)
            features.append(x)
            labels.append(int(label) - 1)
        batches[batch] = np.array(features), np.array(labels, dtype=np.int64)
    return batches


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
