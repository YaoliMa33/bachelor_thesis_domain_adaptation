"""Validate saved experiments and report equal-task and sample-weighted accuracy."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import csv
import io
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from params import DATASET, METHODS, Config
from loaddataset import tasks, load_data_with_fingerprint
from utils import load_result
from experiment_io import (check_provenance, validate_scores, validate_run,
                           summarize_accuracies, atomic_write, runtime_provenance)

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "st_dataset/st_54d/features_54d.csv"


@torch.no_grad()
def evaluate(model, batches, evaluation_batches, arrays):
    """Evaluate the final model on already source-normalized target arrays."""
    model.eval()
    scores = {}
    for batch, x in zip(evaluation_batches, arrays, strict=True):
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite evaluation logits.")
        prediction = logits.argmax(1).numpy()
        truth = batches[batch][1]
        if len(prediction) != len(truth):
            raise ValueError("Prediction and label counts differ.")
        scores[str(batch)] = dict(accuracy=float(100 * (prediction == truth).mean()),
                                  n=len(truth), correct=int(np.count_nonzero(prediction == truth)),
                                  predictions=prediction.tolist())
    return scores


def summarize(output, dataset, seeds, data=None):
    config = Config()
    validate_run(config, seeds, METHODS)
    batches, fingerprint = load_data_with_fingerprint(DEFAULT_DATA if data is None else data)
    values, counts = defaultdict(list), defaultdict(list)
    provenance = None
    task_list = tasks()
    for seed in seeds:
        for task in task_list:
            for method in METHODS:
                file = Path(output) / dataset / f"seed_{seed:02d}" / task.name / f"{method}.json"
                expected = replace(config, lc_weight=config.lc_weight if method.endswith("LC") else 0.0)
                record = load_result(file, dataset=dataset, method=method, seed=seed,
                                     task=asdict(task), config=asdict(expected), data=fingerprint,
                                     entropy_conditioning=False)
                provenance = check_provenance(record, file, fingerprint, provenance)
                validate_scores(record, batches, task.evaluations, file)
                for setting, batch in task.memberships:
                    score = record["scores"][str(batch)]
                    values[setting, method, seed].append(score["accuracy"])
                    counts[setting, method, seed].append((score["correct"], score["n"]))
    settings = list(dict.fromkeys(setting for task in task_list for setting, _ in task.memberships))
    summary = {}
    for setting in settings:
        raw = {m: [statistics.mean(values[setting, m, s]) for s in seeds] for m in METHODS}
        weighted = {m: [100 * sum(c for c, n in counts[setting, m, s])
                        / sum(n for c, n in counts[setting, m, s]) for s in seeds] for m in METHODS}
        summary[setting] = dict(summarize_accuracies(raw), sample_weighted=summarize_accuracies(weighted))
    return dict(result="A", dataset=dataset, seeds=seeds, settings=summary,
                accuracy_unit="percent", gain_unit="percentage_points",
                training_provenance=provenance, evaluation_provenance=runtime_provenance(fingerprint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="The same input data used for training; checked by SHA-256.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    args = parser.parse_args()
    try:
        report = summarize(args.output, DATASET, args.seeds, args.data)
        folder = args.output / DATASET
        pairs = [f for f in ("DANN", "CDAN") if f in METHODS and f + "LC" in METHODS]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["setting", *METHODS, *[f + "_LC_gain_pp" for f in pairs],
                         *[m + "_sample_weighted" for m in METHODS],
                         *[f + "_sample_weighted_LC_gain_pp" for f in pairs]])
        for setting, row in report["settings"].items():
            weighted = row["sample_weighted"]
            writer.writerow([setting, *[row["accuracy"][m]["mean"] for m in METHODS],
                             *[row["paired_LC_gains"][f]["mean"] for f in pairs],
                             *[weighted["accuracy"][m]["mean"] for m in METHODS],
                             *[weighted["paired_LC_gains"][f]["mean"] for f in pairs]])
        atomic_write(folder / "summary.json", json.dumps(report, indent=2, allow_nan=False).encode("utf-8"))
        atomic_write(folder / "summary.csv", stream.getvalue().encode("utf-8-sig"))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
