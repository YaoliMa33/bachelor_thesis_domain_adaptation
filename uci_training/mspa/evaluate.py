"""Average probabilities from the fixed 20-seed pool and summarize MSPA."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parent.parent / "dann_cdan_lc"
sys.path.insert(0, str(SHARED))
import numpy as np
import torch
from loaddataset import load_data, preprocess, tasks
from model import build_model
from params import Config, METHODS

SEEDS = tuple(range(20))
OFFSETS = (0, 7, 13, 3, 17, 5, 11, 1, 9, 15, 2, 12, 4, 8, 16, 6, 10, 14, 18, 19)


def seed_groups(size):
    """Use the thesis cyclic groups; all 20 members form one unique ensemble."""
    return [sorted((g + offset) % 20 for offset in OFFSETS[:size])
            for g in range(1 if size == 20 else 20)]


@torch.no_grad()
def member_probabilities(folder, method, seed, target_x):
    record = json.loads((folder / f"{method}.json").read_text(encoding="utf-8"))
    model = build_model(method, Config(**record["config"]), seed)
    model.load_state_dict(torch.load(folder / f"{method}.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model(target_x).softmax(1).numpy()


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
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    batches = load_data(args.data)
    groups = {size: seed_groups(size) for size in args.sizes}
    rows = []
    for task in tasks(args.uci_protocol):
        target = task.evaluations[0]
        target_x, = preprocess(batches[task.source][0], [batches[target][0]])
        labels = batches[target][1]
        for method in METHODS:
            probabilities = np.stack([
                member_probabilities(args.members / "UCI" / f"seed_{seed:02d}" / task.name,
                                     method, seed, target_x) for seed in SEEDS])
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
                  accuracy_unit="percent",
                  aggregation="Equal task mean per group, then mean and sample SD across groups.",
                  note="Groups overlap. Size 20 is one ensemble and has no across-group SD.",
                  summary=summary)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
