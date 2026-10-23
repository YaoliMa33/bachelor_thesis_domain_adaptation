# UCI Training

[Environment setup](../README.md#environment) · [Dataset guide](../uci_dataset_128d/README.md)

| Module | Role |
|---|---|
| [dann_cdan_lc](dann_cdan_lc) | Shared DANN, DANNLC, CDAN, and CDANLC implementation |
| [entropy/train.py](entropy/train.py) | Entropy conditioning (+E) |
| [mspa](mspa/README.md) | Multi-seed training and probability averaging |

## Data setup

Run `python uci_dataset_128d/download.py` from the repository root before training. All modules use `uci_dataset_128d/data/` by default; see [data source and terms](../uci_dataset_128d/README.md).

## Base models

From the repository root, train all four models with seeds 0–19, then summarize:

```bash
python uci_training/dann_cdan_lc/main.py
python uci_training/dann_cdan_lc/evaluate.py
```

Append `--seeds 0 1` to both commands for a shorter first run with unchanged update counts. `main.py` accepts `--data PATH` for input files; both commands accept `--output PATH` for results. Select models and parameters in [params.py](dann_cdan_lc/params.py), e.g. `METHODS = ("CDANLC",)`.

Choose `--uci-protocol long|medium|both` (default: `both`):

- **Setting B, long-term:** B1 → B2,…,B10, nine tasks.
- **Setting C, medium-term:** Bk → B(k+1), k=1,…,9, including B5 → B6.

Each task starts from a fresh initialization and source-fitted standardization. With `both`, B1→B2 is trained once and counted in both settings. Training and summarization must use matching paths, seeds and protocols. Target labels are used only for evaluation, never for backpropagation or checkpoint selection.

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
| Source sampling | Stratified cycling: proportional allocation, at least one sample per present class |
| Target sampling | Continuous across intervals, carrying over incomplete batches |

An `epoch` records ten updates, not a full data pass. Preprocessing is asinh followed by source-fitted z-score standardization. Initialization is reset with the seed after network construction. DANN feeds features to the domain classifier; CDAN feeds their outer product with detached class probabilities. LC aligns source centers with target-probability-weighted centers for classes present in the source minibatch. Domain weights are projected to the spectral cap at initialization and after each update.

The shared files are `main.py` (task loop), `loaddataset.py` (inputs and protocols), `params.py` (settings), `model.py` (networks), `train.py` (optimization), `evaluate.py` (evaluation and summaries), and `utils.py` (sampling, validation and saving).

## Entropy conditioning (+E)

Supports all four models, using the shared parameters and the long-term protocol by default:

```bash
python uci_training/entropy/train.py
python uci_training/dann_cdan_lc/evaluate.py --output uci_training/entropy/results --uci-protocol long --entropy-conditioning
```

Domain weights use `H = -sum(p * log(p + 1e-5))` and `w = 1 + exp(-H)`. Source and target weights are normalized separately with detached denominators; the entropy branch uses the same GRL coefficient. This follows the [CDAN implementation](https://github.com/thuml/CDAN/blob/master/pytorch/loss.py); applying +E to DANN is a project extension.

Records under `uci_training/entropy/results/UCI/` identify +E through `entropy_conditioning` and `variant`. LC gains compare models that both use +E, not the effect of adding E itself.

## Multi-seed probability averaging (MSPA)

```bash
python uci_training/mspa/train.py
python uci_training/mspa/evaluate.py
```

MSPA averages base-model probabilities without +E. It requires seeds 0–19 for each model and task; default groups contain seven members. See the [MSPA guide](mspa/README.md) for checkpoint reuse, group definitions and ensemble-size comparisons.

## Outputs and statistics

Base-model JSON records and final `.pt` checkpoints are saved under `uci_training/dann_cdan_lc/results/UCI/`, grouped by seed, task and model. `evaluate.py` summarizes them without retraining and writes `summary.json` and `summary.csv` in that directory.

Saving and summarization reject mismatched configurations, tasks, seeds or variants; summarization checks the current `params.py`. Matching reruns overwrite corresponding files. Use a new output directory for a new experiment; default results and checkpoints are ignored by Git.

For each setting and seed, accuracy (%) is averaged equally over nine tasks; across seeds, report mean and sample SD (`ddof=1`, `null` for one seed). JSON includes seed means, mean/SD and paired LC gains; CSV contains means and mean gains. Gains are in percentage points and require both the base and LC model in a family.
