"""Train and evaluate result A with user-supplied data."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from params import DATASET, METHODS, Config
from loaddataset import load_data, preprocess, tasks
from train import train
from test import evaluate
from utils import save_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parents[2] / "st_dataset/st_54d/features_54d.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    args = parser.parse_args()
    batches = load_data(args.data)
    config = Config()

    for seed in args.seeds:
        for task in tasks():
            source_x, source_y = batches[task.source]
            target_x = np.concatenate([batches[b][0] for b in task.targets])
            arrays = preprocess(source_x, [source_x, target_x]
                                + [batches[b][0] for b in task.evaluations])
            for method in METHODS:
                weight = config.lc_weight if method.endswith("LC") else 0.0
                method_config = replace(config, lc_weight=weight)
                model, history = train(method, arrays[0], torch.from_numpy(source_y),
                                       arrays[1], method_config, seed)
                scores = evaluate(model, batches, task.evaluations, arrays[2:])
                record = dict(result="A", dataset=DATASET, method=method,
                              seed=seed, task=asdict(task), config=asdict(method_config),
                              scores=scores, history=history)
                folder = args.output / DATASET / f"seed_{seed:02d}" / task.name
                save_result(folder, method, model, record)
                print(f"{DATASET} seed={seed} {task.name} {method}", flush=True)


if __name__ == "__main__":
    main()
