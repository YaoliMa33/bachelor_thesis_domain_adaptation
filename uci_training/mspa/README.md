# Multi-Seed Probability Averaging (MSPA)

This folder contains only the MSPA entry points. The DANN, DANNLC, CDAN and CDANLC implementations remain in `../dann_cdan_lc/` and are imported directly. MSPA adds no training loss and does not use the entropy-conditioned trainer in `../entropy/`.

| File | Purpose |
|---|---|
| `train.py` | Run the shared trainer independently for seeds 0–19; save members in this folder's `results/` by default. |
| `evaluate.py` | Load final member checkpoints, average softmax probabilities, and summarize accuracy across the fixed seed groups. |

## Training and evaluation

Run from the repository root. Dependencies are the same as the shared UCI trainer.

```bash
# Train both Setting B (long) and Setting C (medium), all four methods, seeds 0–19.
python uci_training/mspa/train.py

# Evaluate the thesis reference ensemble size M=7.
python uci_training/mspa/evaluate.py

# Ensemble-size sensitivity; no additional training is needed.
python uci_training/mspa/evaluate.py --sizes 1 3 5 7 10 12 15 20 --output uci_training/mspa/results/size_sensitivity

# Reuse an existing pool produced by the shared trainer instead of training again.
python uci_training/mspa/evaluate.py --members uci_training/dann_cdan_lc/results --uci-protocol long
```

Both entry points accept `--data`, `--output` and `--uci-protocol long|medium|both`. The training entry also accepts `--seeds`, so the pool may be trained in separate invocations; evaluation requires all 20 seeds (0–19) for every selected method and task. `--members` selects the training output root, not its `UCI/` child. With a custom training output, pass that same path as `--members`. Set `METHODS` and training parameters in `../dann_cdan_lc/params.py`. Each member JSON stores the actual configuration; evaluation reconstructs its model using that saved configuration.

The retained training defaults are 450 updates, domain weight 0.8, and LC weight 0.2 for LC models (zero for DANN/CDAN). Preprocessing, architecture, initialization, sampling, optimizer and GRL scheduling are unchanged. Setting B uses B1→B2,…,B10; Setting C uses B1→B2,…,B9→B10, including B5→B6. With `both`, the identical B1→B2 training task is run once and evaluated in both settings. No checkpoint or ensemble member is selected by target accuracy.

## Probability averaging and groups

For each target sample, MSPA takes the arithmetic mean of the members' full softmax vectors and then selects the largest probability. It does not average logits, model parameters or hard predicted labels. Each ensemble contains only one model type.

The fixed seed pool is 0–19. For size M and group g, take the first M offsets below and use `(g + offset) % 20`:

```text
0, 7, 13, 3, 17, 5, 11, 1, 9, 15, 2, 12, 4, 8, 16, 6, 10, 14, 18, 19
```

For M<20, g=0,…,19 gives 20 groups, shared across all models and tasks. Each seed occurs in M groups. For M=20 there is one unique group. Member IDs are sorted before summation, and averaging uses float64, matching the experimental analysis. The default is M=7; M=1 recovers individual-seed evaluation.

Accuracy is first averaged equally over the nine tasks within each group, then summarized over groups as mean ± sample SD (`ddof=1`). The groups overlap: their SD describes ensemble-composition variation within the fixed seed pool, not independent ensemble runs or a confidence interval. M=20 has no across-group SD (`null` in JSON, blank in CSV). The reference size was chosen through exploratory comparisons; this script does not select a new optimum.

## Generated files

- Member training: `results/UCI/seed_00/B1_B2/DANN.json` and `DANN.pt`, with corresponding paths for every seed, task and model.
- `results/mspa/per_task.csv`: accuracy for each task, method, ensemble size and group.
- `results/mspa/per_group.csv`: task-equal accuracy for each group and setting.
- `results/mspa/summary.csv`: mean accuracy and sample SD across groups.
- `results/mspa/summary.json`: the same statistics plus exact seed groups and aggregation settings.

Results and checkpoints are generated locally and are not included in this source folder. Re-running into the same output path replaces files with the same names; use separate output folders for different training configurations. For preprocessing and shared training details, see the [UCI training guide](../README.md); labels and provenance are in the [UCI dataset guide](../../uci_dataset_128d/README.md).
