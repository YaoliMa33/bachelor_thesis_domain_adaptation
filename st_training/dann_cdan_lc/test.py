"""Model accuracy evaluation and equal-batch, equal-seed result reporting."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import torch

from params import DATASET, METHODS
from loaddataset import tasks


@torch.no_grad()
def evaluate(model, batches, evaluation_batches, arrays):
    """Evaluate a trained model on the already source-normalized target arrays."""
    model.eval()
    scores = {}
    for batch, x in zip(evaluation_batches, arrays):
        prediction = model(x).argmax(1).numpy()
        truth = batches[batch][1]
        scores[str(batch)] = dict(
            accuracy=float(100 * (prediction == truth).mean()),
            n=len(truth), predictions=prediction.tolist())
    return scores


def summarize(output, dataset, seeds):
    values = defaultdict(list)
    task_list = tasks()
    for seed in seeds:
        for task in task_list:
            for method in METHODS:
                file = output / dataset / f"seed_{seed:02d}" / task.name / f"{method}.json"
                record = json.loads(file.read_text(encoding="utf-8"))
                for setting, batch in task.memberships:
                    values[setting, method, seed].append(record["scores"][str(batch)]["accuracy"])
    settings = list(dict.fromkeys(setting for task in task_list
                                  for setting, batch in task.memberships))
    summary = {}
    for setting in settings:
        raw = {method: [statistics.mean(values[setting, method, s]) for s in seeds] for method in METHODS}
        accuracy = {m: dict(mean=statistics.mean(v), sd=statistics.stdev(v) if len(v) > 1 else None) for m, v in raw.items()}
        paired = {}
        for family in (f for f in ("DANN", "CDAN") if f in raw and f + "LC" in raw):
            differences = [b - a for a, b in zip(raw[family], raw[family + "LC"])]
            paired[family] = dict(mean=statistics.mean(differences), sd=statistics.stdev(differences) if len(differences) > 1 else None,
                                   positive_seeds=sum(d > 0 for d in differences))
        summary[setting] = dict(accuracy=accuracy, paired_LC_gains=paired, seed_means=raw)
    return dict(result="A", dataset=dataset, seeds=seeds, settings=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    args = parser.parse_args()
    report = summarize(args.output, DATASET, args.seeds)
    folder = args.output / DATASET
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        pairs = [f for f in ("DANN", "CDAN") if f in METHODS and f + "LC" in METHODS]
        writer.writerow(["setting", *METHODS, *[f + "_LC_gain_pp" for f in pairs]])
        for setting, row in report["settings"].items():
            writer.writerow([setting, *[row["accuracy"][m]["mean"] for m in METHODS],
                             *[row["paired_LC_gains"][f]["mean"] for f in pairs]])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
