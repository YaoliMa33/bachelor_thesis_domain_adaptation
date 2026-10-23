# Multi-Seed Probability Averaging (MSPA)

[UCI training guide](../README.md) · [Dataset guide](../../uci_dataset_128d/README.md)

MSPA averages predictions from independently trained models of the same type. Training is imported from `../dann_cdan_lc`; no extra loss or entropy conditioning is added.

| File | Role |
|---|---|
| train.py | Train the shared models with seeds 0–19 |
| evaluate.py | Load checkpoints, average probabilities, and summarize accuracy |

## Run

From the repository root:

```bash
python uci_training/mspa/train.py
python uci_training/mspa/evaluate.py

# Compare ensemble sizes using the same trained members.
python uci_training/mspa/evaluate.py --sizes 1 3 5 7 10 12 15 20 --output uci_training/mspa/results/size_sensitivity

# Reuse existing base-model checkpoints.
python uci_training/mspa/evaluate.py --members uci_training/dann_cdan_lc/results --uci-protocol long
```

Both entry points accept `--data`, `--output`, and `--uci-protocol long|medium|both`. Training also accepts `--seeds`; evaluation requires the complete pool 0–19 for each selected model and task. `--members` points to the training output root, not its `UCI` subfolder. Select models in [params.py](../dann_cdan_lc/params.py); checkpoint JSON records are checked against the current parameters, task, model and seed before loading. Base models and +E records cannot be mixed. Missing records or checkpoints stop evaluation with an explicit error.

Defaults are 450 training updates, domain weight 0.8, LC weight 0.2 for LC models, and both protocols. All nine medium-term tasks are included. The shared B1 → B2 task is trained once. No checkpoint or member is selected by target accuracy.

## Averaging and seed groups

For each sample, average the members’ softmax vectors in float64, then take the class with the largest probability. Member IDs are sorted before summation. The default group size is seven; size one recovers individual-seed evaluation.

For size M and group g, take the first M offsets below and use `(g + offset) % 20`:

```text
0, 7, 13, 3, 17, 5, 11, 1, 9, 15, 2, 12, 4, 8, 16, 6, 10, 14, 18, 19
```

For M<20, g=0,…,19 gives 20 groups, shared across models and tasks; each seed appears in M groups. M=20 gives one unique ensemble.

Average accuracy equally across the nine tasks within each group, then report the group mean and sample SD (`ddof=1`). Groups overlap, so SD describes composition variation within the fixed seed pool, not independent repeated experiments or a confidence interval. For M=20, SD is `null` in JSON and blank in CSV.

## Outputs

Member checkpoints and training records are saved under `results/UCI/seed_00/B1_B2/`, with corresponding paths for all seeds, tasks, and models. Evaluation writes the following files under `results/mspa/`:

- `per_task.csv`: accuracy by task, model, ensemble size, and group.
- `per_group.csv`: task-averaged accuracy for each group and setting.
- `summary.csv`: group mean accuracy and sample SD.
- `summary.json`: summary statistics, exact seed groups, and aggregation settings.

Reusing an output path overwrites matching files. Use separate directories for different configurations. Results and checkpoints are not included in the repository.
