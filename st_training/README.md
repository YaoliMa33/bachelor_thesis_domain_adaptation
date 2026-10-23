# ST Training

[Environment setup](../README.md#environment) · [Dataset guide](../st_dataset/README.md)

`dann_cdan_lc` implements DANN, DANNLC, CDAN, and CDANLC for the three protocols of Setting A.

## Run

From the repository root, train all four models with seeds 0–19, then summarize:

```bash
python st_training/dann_cdan_lc/main.py
python st_training/dann_cdan_lc/evaluate.py
```

For a shorter first run, append `--seeds 0 1` to both commands; update counts remain unchanged. `main.py` accepts `--data PATH` for the feature CSV. Both commands accept `--output PATH` and must use the same output path and seeds. Select models and parameters in [params.py](dann_cdan_lc/params.py), e.g. `METHODS = ("CDANLC",)`.

## Protocols and defaults

B1 is always the labeled source. Target labels are used only for evaluation, never for backpropagation or checkpoint selection.

| Protocol | Unlabeled target pool | Evaluation |
|---|---|---|
| Protocol 1 / Setting1 | B2–B6 pooled | Each of B2–B6 |
| Protocol 2 / Setting2 | Current batch Bk | Bk, k=2,…,6 |
| Protocol 3 / Setting3 | B2,…,Bk | Bk, k=2,…,6 |

Protocol 1 trains one model per seed and method, using the same target samples for adaptation and evaluation. Protocols 2 and 3 reinitialize for each task; Protocol 3 accumulates data, not model parameters. Their identical B1→B2 task is trained once and counted in both settings, giving ten distinct training tasks overall.

| Parameter | Default |
|---|---|
| Input / classes / feature dimension | 54 / 3 / 200 |
| Feature extractor | Linear → BatchNorm → ReLU; joint source–target BN |
| Domain hidden layers / spectral cap | 32 → 32 / 0.75 |
| Minibatch per domain / updates | 20 / 300 (60 × 5) |
| Optimizer / learning rate / weight decay | Adam / 0.005, fixed / 0.0001 |
| Domain / LC weight | 0.9 / 1.2; LC is zero for DANN and CDAN |
| GRL | gamma=3; increases for 150 updates, then stays fixed |
| Source sampling | Stratified cycling: 4 air, 8 alcohol, 8 acetone per minibatch |
| Target sampling | Shuffled loader, restarted each recording interval; incomplete batches dropped |

An `epoch` records five updates, not a full data pass; optimizer state persists. Source classes cycle through shuffled pools. `source_class_counts = (4, 8, 8)` follows air, alcohol, acetone and must sum to `batch_size`.

Preprocessing is asinh followed by source-fitted z-score standardization. Initialization is reset with the seed after network construction. DANN feeds features to the domain classifier; CDAN feeds their outer product with detached class probabilities. LC aligns source centers with target-probability-weighted centers for classes present in the source minibatch. Domain weights are projected to the spectral cap at initialization and after each update.

## Outputs and statistics

JSON records and final `.pt` checkpoints are saved under `st_training/dann_cdan_lc/results/ST/`, grouped by seed, task and model. `evaluate.py` summarizes these records without retraining and writes `summary.json` and `summary.csv` in that directory.

Saving and summarization reject mismatched configurations, tasks, seeds or variants; summarization checks the current `params.py`. Matching reruns overwrite corresponding files. Use a new output directory for a new experiment; default results and checkpoints are ignored by Git.

Within each seed, accuracy (%) is averaged equally over five target batches; across seeds, report mean and sample SD (`ddof=1`, `null` for one seed). JSON includes seed means, mean/SD and paired LC gains; CSV contains means and mean gains. Gains are in percentage points and require both the base and LC model in a family.

## Files

| File | Role |
|---|---|
| main.py | Experiment entry point and task loop |
| loaddataset.py | Data loading, preprocessing, and protocols |
| params.py | Parameters and model selection |
| model.py | Networks, initialization, GRL, and spectral projection |
| train.py | Losses and optimization |
| evaluate.py | Accuracy evaluation and result summaries |
| utils.py | Sampling, record validation, and saving |
