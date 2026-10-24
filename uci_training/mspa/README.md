# Multi-Seed Probability Averaging (MSPA)

[UCI training guide](../README.md) · [Dataset guide](../../uci_dataset_128d/README.md)

MSPA averages predictions from independently trained seeds of the same model. `train.py` reuses the [base UCI trainer and defaults](../README.md#shared-defaults), without +E; `evaluate.py` loads checkpoints and averages their probabilities.

## Run

From the repository root, after UCI data setup:

```bash
python uci_training/mspa/train.py
python uci_training/mspa/evaluate.py

# Compare sizes using the same trained members.
python uci_training/mspa/evaluate.py --sizes 1 3 5 7 10 12 15 20 --output uci_training/mspa/results/size_sensitivity

# Reuse existing base-model checkpoints.
python uci_training/mspa/evaluate.py --members uci_training/dann_cdan_lc/results --uci-protocol long
```

Both commands accept `--data`, `--output` and `--uci-protocol long|medium|both` (default: `both`). Training accepts `--seeds`; evaluation requires all seeds 0–19 for every selected model and task. `--members` is the training output root, not its `UCI` subfolder. Models are selected in [params.py](../dann_cdan_lc/params.py).

Evaluation checks the current data hashes, parameters, task, model, seed and saved predictions, requires consistent training provenance, and verifies each checkpoint against its JSON hash. Missing files or mismatches stop evaluation; base and +E records cannot be mixed. Old records without provenance require retraining in a new output directory. No member or checkpoint is selected by target accuracy.

## Averaging and seed groups

For each sample, average members' softmax vectors in float64, then select the largest probability. Members are sorted by seed before summation. Default size is seven; size one recovers individual-seed evaluation.

For size M and group g, take the first M offsets below and compute `(g + offset) % 20`:

```text
0, 7, 13, 3, 17, 5, 11, 1, 9, 15, 2, 12, 4, 8, 16, 6, 10, 14, 18, 19
```

For M<20, g=0,…,19 gives 20 groups shared across models and tasks; each seed appears in M groups. M=20 gives one unique ensemble.

For each setting, average accuracy (%) equally over nine tasks per group, then report group mean and sample SD (`ddof=1`). Groups overlap: SD describes composition variation within the fixed seed pool, not independent experiments or a confidence interval. For M=20, SD is `null` in JSON and blank in CSV.

## Outputs

Member records and checkpoints follow `uci_training/mspa/results/UCI/seed_00/B1_B2/`. Evaluation writes to `uci_training/mspa/results/mspa/` by default:

| File | Contents |
|---|---|
| per_task.csv | Accuracy by task, model, size and group |
| per_group.csv | Task-averaged accuracy per group and setting |
| summary.csv | Group mean and sample SD |
| summary.json | Statistics, exact groups and aggregation settings |

Matching output files are overwritten; use separate directories for different configurations. Results and checkpoints are not bundled.
