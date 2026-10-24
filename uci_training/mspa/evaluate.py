"""Average probabilities from the fixed 20-seed pool and summarize MSPA."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import csv
import io
import json
from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parent.parent / "dann_cdan_lc"
sys.path.insert(0, str(SHARED))
import numpy as np
import torch
from loaddataset import load_data_with_fingerprint, preprocess, tasks
from model import build_model
from params import Config, METHODS
from utils import load_result
from experiment_io import (check_provenance, validate_scores, validate_checkpoint,
                           runtime_provenance, atomic_write, validate_run)

SEEDS = tuple(range(20))
OFFSETS = (0, 7, 13, 3, 17, 5, 11, 1, 9, 15, 2, 12, 4, 8, 16, 6, 10, 14, 18, 19)


def seed_groups(size):
    """Use the thesis cyclic groups; all 20 members form one unique ensemble."""
    if type(size) is not int or not 1 <= size <= 20:
        raise ValueError("MSPA size must be an integer from 1 to 20.")
    return [sorted((g + offset) % 20 for offset in OFFSETS[:size])
            for g in range(1 if size == 20 else 20)]


@torch.no_grad()
def member_probabilities(folder, method, seed, target_x, task, data, batches, provenance=None):
    config = Config()
    config = replace(config, lc_weight=config.lc_weight if method.endswith("LC") else 0.0)
    file = folder / f"{method}.json"
    record = load_result(file, dataset="UCI", method=method, seed=seed,
                         task=asdict(task), config=asdict(config), data=data)
    provenance = check_provenance(record, file, data, provenance)
    validate_scores(record, batches, task.evaluations, file)
    content = validate_checkpoint(folder / f"{method}.pt", record)
    model = build_model(method, Config(**record["config"]), seed)
    model.load_state_dict(torch.load(io.BytesIO(content), map_location="cpu", weights_only=True))
    model.eval()
    probabilities = model(target_x).softmax(1)
    if not torch.isfinite(probabilities).all():
        raise ValueError(f"Non-finite member predictions: {file}")
    return probabilities.numpy(), provenance


def ensemble_accuracies(probabilities, labels, groups):
    """Average complete softmax vectors in float64, then select the class."""
    return [float(100 * (probabilities[members].astype(np.float64).mean(0)
                        .argmax(1) == labels).mean()) for members in groups]


def summarize(rows):
    by_group = defaultdict(list)
    for row in rows:
        key = (row["setting"], row["method"], row["size"], row["group"])
        by_group[key].append(row["accuracy"])
    group_rows = [dict(setting=s, method=m, size=n, group=g,
                       accuracy=float(np.mean(values)))
                  for (s, m, n, g), values in by_group.items()]
    by_setting = defaultdict(list)
    for row in group_rows:
        by_setting[row["setting"], row["method"], row["size"]].append(row["accuracy"])
    summary = [dict(setting=s, method=m, size=n, groups=len(values),
                    mean=float(np.mean(values)),
                    sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
               for (s, m, n), values in by_setting.items()]
    return group_rows, summary


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue().encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parents[2] / "uci_dataset_128d/data")
    parser.add_argument("--members", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="Training output root containing UCI/seed_00/... through seed_19.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "mspa")
    parser.add_argument("--uci-protocol", choices=["long", "medium", "both"], default="both")
    parser.add_argument("--sizes", type=int, nargs="+", choices=range(1, 21), default=[7])
    args = parser.parse_args()
    torch.set_num_threads(1)
    validate_run(Config(), list(SEEDS), METHODS)
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("Select distinct ensemble sizes.")
    batches, fingerprint = load_data_with_fingerprint(args.data)
    provenance = None
    groups = {size: seed_groups(size) for size in args.sizes}
    rows = []
    for task in tasks(args.uci_protocol):
        target = task.evaluations[0]
        target_x, = preprocess(batches[task.source][0], [batches[target][0]])
        labels = batches[target][1]
        for method in METHODS:
            predictions = []
            for seed in SEEDS:
                probabilities, provenance = member_probabilities(
                    args.members / "UCI" / f"seed_{seed:02d}" / task.name,
                    method, seed, target_x, task, fingerprint, batches, provenance)
                predictions.append(probabilities)
            probabilities = np.stack(predictions)
            for size, members in groups.items():
                accuracies = ensemble_accuracies(probabilities, labels, members)
                for setting, batch in task.memberships:
                    for group, accuracy in enumerate(accuracies):
                        rows.append(dict(setting=setting, method=method, size=size,
                                         group=group, task=task.name, target=batch,
                                         accuracy=accuracy))
            print(f"{task.name} {method}: MSPA evaluated", flush=True)
    group_rows, summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_task.csv", rows)
    write_csv(args.output / "per_group.csv", group_rows)
    write_csv(args.output / "summary.csv", summary)
    report = dict(seeds=SEEDS, offsets=OFFSETS, groups=groups,
                  methods=METHODS, uci_protocol=args.uci_protocol,
                  accuracy_unit="percent", training_provenance=provenance,
                  evaluation_provenance=runtime_provenance(fingerprint),
                  aggregation="Equal task mean per group, then mean and sample SD across groups.",
                  note="Groups overlap. Size 20 is one ensemble and has no across-group SD.",
                  summary=summary)
    atomic_write(args.output / "summary.json", json.dumps(report, indent=2, allow_nan=False).encode("utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"MSPA evaluation failed: {exc}") from None
