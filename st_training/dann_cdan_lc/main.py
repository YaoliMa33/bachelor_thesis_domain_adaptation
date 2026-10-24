"""Train and evaluate ST domain-adaptation experiments."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from params import DATASET, METHODS, Config
from loaddataset import load_data_with_fingerprint, preprocess, tasks
from train import train
from evaluate import evaluate
from utils import save_result
from experiment_io import SCHEMA_VERSION, runtime_provenance, validate_run, check_existing


def main(entropy_conditioning=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parents[2] / "st_dataset/st_54d/features_54d.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    args = parser.parse_args()
    try:
        batches, fingerprint = load_data_with_fingerprint(args.data)
        config = Config()
        validate_run(config, args.seeds, METHODS)
        provenance = runtime_provenance(fingerprint)
        jobs = []
        for seed in args.seeds:
            for task in tasks():
                source_size = len(batches[task.source][1])
                target_size = sum(len(batches[b][1]) for b in task.targets)
                if ((config.source_sampling == "loader" and source_size < config.batch_size)
                        or (config.target_sampling == "loader" and target_size < config.batch_size)):
                    raise ValueError(f"{task.name} has too few samples for batch_size with drop_last enabled.")
                for method in METHODS:
                    method_config = replace(config, lc_weight=config.lc_weight if method.endswith("LC") else 0.0)
                    record = dict(schema_version=SCHEMA_VERSION, result="A", dataset=DATASET,
                                  method=method, seed=seed, task=asdict(task), config=asdict(method_config),
                                  entropy_conditioning=entropy_conditioning,
                                  variant=method + ("+E" if entropy_conditioning else ""), provenance=provenance)
                    folder = args.output / DATASET / f"seed_{seed:02d}" / task.name
                    check_existing(folder, method, record)
                    jobs.append((seed, task, method, method_config, folder, record))
        # Check every existing destination before starting any training or writes.
        for seed, task, method, method_config, folder, record in jobs:
            source_x, source_y = batches[task.source]
            target_x = np.concatenate([batches[b][0] for b in task.targets])
            arrays = preprocess(source_x, [source_x, target_x]
                                + [batches[b][0] for b in task.evaluations])
            model, history = train(method, arrays[0], torch.from_numpy(source_y),
                                   arrays[1], method_config, seed)
            scores = evaluate(model, batches, task.evaluations, arrays[2:])
            record = dict(record, scores=scores, history=history)
            save_result(folder, method, model, record)
            print(f"{DATASET} seed={seed} {task.name} {method}", flush=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
