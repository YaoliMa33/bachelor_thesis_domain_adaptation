"""Mini-batch sampling and experiment-result saving."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_io import save_result, load_result
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Cycle:
    def __init__(self, indices, seed):
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or not len(self.indices):
            raise ValueError("Cannot cycle an empty or non-vector sample pool.")
        self.rng = np.random.RandomState(seed)
        self.pool = np.empty(0, dtype=np.int64)
        self.position = 0

    def take(self, count):
        if not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Sample count must be a nonnegative integer.")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        chunks = []
        while count:
            if self.position >= len(self.pool):
                self.pool = self.indices[self.rng.permutation(len(self.indices))]
                self.position = 0
            size = min(count, len(self.pool) - self.position)
            chunks.append(self.pool[self.position:self.position + size])
            self.position += size
            count -= size
        return np.concatenate(chunks)


class Stratified:
    def __init__(self, labels, batch_size, seed):
        labels = np.asarray(labels)
        if labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Stratified sampling requires a nonempty integer label vector.")
        classes, counts = np.unique(labels, return_counts=True)
        if type(batch_size) is not int or batch_size < len(classes):
            raise ValueError("batch_size must accommodate at least one sample per present class.")
        ideal = counts * batch_size / counts.sum()
        allocation = np.maximum(1, np.floor(ideal).astype(int))
        while allocation.sum() < batch_size:
            allocation[np.argmax(ideal - allocation)] += 1
        while allocation.sum() > batch_size:
            eligible = np.where(allocation > 1, allocation - ideal, -np.inf)
            allocation[np.argmax(eligible)] -= 1
        self.counts = allocation
        self.samplers = [Cycle(np.flatnonzero(labels == c), seed + int(c) * 97)
                         for c in classes]
        self.rng = np.random.RandomState(seed + 911)

    def take(self):
        indices = np.concatenate([s.take(int(n)) for s, n in zip(self.samplers, self.counts)])
        return indices[self.rng.permutation(len(indices))]


def make_loaders(source_x, source_y, target_x, config, seed):
    source = DataLoader(TensorDataset(source_x, source_y), batch_size=config.batch_size,
                        shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed + 11000))
    target = DataLoader(TensorDataset(target_x), batch_size=config.batch_size,
                        shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed + 12000))
    return source, target


def repeat(loader):
    if len(loader) == 0:
        raise ValueError("Cannot repeat an empty loader; provide enough samples for batch_size or disable drop_last")
    while True:
        yield from loader


def continuous_target(loader):
    full = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=True,
                      drop_last=False, generator=loader.generator)
    pending = None
    for (x,) in repeat(full):
        pending = x if pending is None else torch.cat((pending, x))
        while len(pending) >= loader.batch_size:
            yield (pending[:loader.batch_size],)
            pending = pending[loader.batch_size:]
