# UCI Training

[Environment setup](../README.md#environment) · [Dataset guide](../uci_dataset_128d/README.md)

| Module | Role |
|---|---|
| [dann_cdan_lc](dann_cdan_lc) | Shared DANN, DANNLC, CDAN, and CDANLC implementation |
| [entropy/train.py](entropy/train.py) | Entropy conditioning (+E), using the shared training loop |
| [mspa](mspa/README.md) | Multi-seed training and probability averaging |

## Data setup

The release does not bundle UCI batch files. First run `python uci_dataset_128d/download.py` from the repository root; all training extensions and MSPA then use the verified local `uci_dataset_128d/data/` directory. See [data source and terms](../uci_dataset_128d/README.md).

## Base models

Run from the repository root. Defaults are all four models and seeds 0–19:

```bash
python uci_training/dann_cdan_lc/main.py
python uci_training/dann_cdan_lc/test.py
```

For an initial run with two seeds and the full training budget:

```bash
python uci_training/dann_cdan_lc/main.py --seeds 0 1
python uci_training/dann_cdan_lc/test.py --seeds 0 1
```

Select `--uci-protocol long|medium|both`; the default is `both`:

- Setting B (long-term): B1 → B2,…,B10, nine tasks.
- Setting C (medium-term): Bk → B(k+1), k=1,…,9, including B5 → B6.

Each task starts from a fresh initialization, with standardization fitted on its source batch. With `both`, B1 → B2 is trained once and included in both settings.

Use `--data PATH` for the batch-file directory, `--output PATH` for an output directory, and `--seeds 0 1` to select seeds. Training and summarization must use matching paths, seeds, and protocols. Select models in [params.py](dann_cdan_lc/params.py), for example `METHODS = ("CDANLC",)`.

## Shared defaults

| Parameter | Default |
|---|---|
| Input / classes / feature dimension | 128 / 6 / 200 |
| Feature extractor | Linear → BatchNorm → ReLU; joint source–target BN |
| Domain hidden layers / spectral cap | 64 → 64 / 1.0 |
| Minibatch per domain / updates | 16 / 450 (45 × 10) |
| Optimizer / initial learning rate | SGD, momentum=0.9 / 0.03; weight decay=0 |
| Learning-rate schedule | Cosine decay per recording interval, eta_min=1e-6 |
| Domain / LC weight | 0.8 / 0.2; LC is zero for DANN and CDAN |
| GRL | gamma=10, scheduled over all 450 updates |
| Source sampling | Stratified cycling, with allocation proportional to class counts |
| Target sampling | Continuous across intervals, carrying over incomplete batches |

An `epoch` is a recording interval of ten updates. Preprocessing applies asinh followed by z-score standardization fitted on the source domain. Initialization is reset using the seed after network construction. DANN uses features as domain inputs; CDAN uses their outer product with detached class probabilities. LC aligns source class centers with centers weighted by target class probabilities, using classes present in the source minibatch. Domain weights are projected onto the spectral-norm constraint at initialization and after each update.

In the shared module, `main.py` manages tasks, `loaddataset.py` loads and preprocesses data, `model.py` defines networks, `train.py` handles optimization, `test.py` evaluates and summarizes, and `utils.py` handles sampling and saving.

## Entropy conditioning (+E)

The extension supports all four models. It defaults to the long-term protocol and uses the shared parameters:

```bash
python uci_training/entropy/train.py
python uci_training/dann_cdan_lc/test.py --output uci_training/entropy/results --uci-protocol long
```

Domain loss is weighted using `H = -sum(p * log(p + 1e-5))` and `w = 1 + exp(-H)`. Source and target weights are normalized separately with detached denominators; the entropy branch uses the same GRL coefficient. This follows the [CDAN implementation](https://github.com/thuml/CDAN/blob/master/pytorch/loss.py); applying +E to DANN is an extension in this project.

Results go to `entropy/results/UCI/`. Records retain the base model name and identify +E through `entropy_conditioning` and `variant`. Reported LC gains compare models that both use +E; they do not measure the gain from adding E.

## Multi-seed probability averaging (MSPA)

```bash
python uci_training/mspa/train.py
python uci_training/mspa/evaluate.py
```

MSPA trains the base models without +E. Evaluation requires seeds 0–19 for every selected model and task. By default, each group contains seven members; their softmax probabilities are averaged before choosing a class. To reuse existing base-model checkpoints, pass `--members uci_training/dann_cdan_lc/results`.

See the [MSPA guide](mspa/README.md) for grouping, ensemble-size comparisons, and output files.

## Outputs and statistics

Base training saves JSON records and final `.pt` checkpoints under `dann_cdan_lc/results/UCI/`, grouped by seed, task, and model. Reusing an output path overwrites matching files. Evaluation uses the final model; `test.py` summarizes saved results without retraining.

Accuracy is averaged equally over the nine tasks within each seed, then summarized across seeds as mean and sample SD (`ddof=1`). JSON includes seed means, mean/SD, and paired LC gains; CSV contains means and mean gains. SD is `null` for one seed. Paired gains require both models in a family.

Target labels do not enter backpropagation or checkpoint selection. Parameters and training budgets were chosen with reference to target performance, so the comparisons are exploratory. Default results, checkpoints, and caches are ignored by Git; use a location outside the repository for custom outputs.
