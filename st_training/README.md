# ST Training

[Environment setup](../README.md#environment) · [Dataset guide](../st_dataset/README.md)

`dann_cdan_lc` implements DANN, DANNLC, CDAN, and CDANLC under the three target-data access protocols of Setting A.

## Run

From the repository root, train all four models with seeds 0–19 and summarize their results:

```bash
python st_training/dann_cdan_lc/main.py
python st_training/dann_cdan_lc/evaluate.py
```

For an initial run with two seeds and the full training budget:

```bash
python st_training/dann_cdan_lc/main.py --seeds 0 1
python st_training/dann_cdan_lc/evaluate.py --seeds 0 1
```

Use `--data PATH` for a feature CSV, `--output PATH` for an output directory, and `--seeds 0 1` to select seeds. Training and summarization must use the same output path and seeds. Set models and parameters in [params.py](dann_cdan_lc/params.py); for example, `METHODS = ("CDANLC",)` selects one model.

## Protocols and defaults

B1 is always the labeled source batch. Target labels are not used for training.

| Protocol | Unlabeled target pool | Evaluation |
|---|---|---|
| Protocol 1 / Setting1 | B2–B6 pooled | Each of B2–B6 |
| Protocol 2 / Setting2 | Current batch Bk | Bk, k=2,…,6 |
| Protocol 3 / Setting3 | B2,…,Bk | Bk, k=2,…,6 |

Protocol 1 trains one model per seed and method. Protocols 2 and 3 reinitialize for each evaluation task. Protocol 3 accumulates observations, not model parameters.

| Parameter | Default |
|---|---|
| Input / classes / feature dimension | 54 / 3 / 200 |
| Feature extractor | Linear → BatchNorm → ReLU; joint source–target BN |
| Domain hidden layers / spectral cap | 32 → 32 / 0.75 |
| Minibatch per domain / updates | 20 / 300 (60 × 5) |
| Optimizer / learning rate / weight decay | Adam / 0.005, fixed / 0.0001 |
| Domain / LC weight | 0.9 / 1.2; LC is zero for DANN and CDAN |
| GRL | gamma=3; increases for 150 updates, then stays fixed |
| Source sampling | Stratified cycling: 4 air, 8 alcohol, and 8 acetone samples per minibatch |
| Target sampling | Shuffled loader, restarted each recording interval; incomplete batches dropped |

An `epoch` is a recording interval of five updates, not a full pass through the data. Optimizer state is retained throughout training. Source samples cycle through a shuffled pool for each class; `source_class_counts = (4, 8, 8)` follows the class order air, alcohol, acetone and must sum to `batch_size`.

Preprocessing applies asinh followed by z-score standardization fitted on the source domain. Initialization is reset using the seed after network construction. DANN uses features as domain inputs; CDAN uses their outer product with detached class probabilities. LC aligns source class centers with centers weighted by target class probabilities, using classes present in the source minibatch. Domain weights are projected onto the spectral-norm constraint at initialization and after each update.

## Outputs and statistics

Training writes JSON records and final `.pt` checkpoints under `dann_cdan_lc/results/ST/`, grouped by seed, task, and model. Saving and summarization reject existing records with different parameters, task identities, seeds or model variants. A rerun with matching settings overwrites matching files; use a new output directory for a new experiment. Summarization validates records against the current `params.py`. Evaluation uses the final model; `evaluate.py` summarizes saved results without retraining.

Accuracy is averaged equally over the five target batches within each seed, then summarized across seeds as mean and sample SD (`ddof=1`). JSON includes seed means, mean/SD, and paired LC gains; CSV contains means and mean gains. SD is `null` for one seed. Paired gains are calculated only when both models in a family are selected.

Target labels do not enter backpropagation or checkpoint selection. Default results, checkpoints, and caches are ignored by Git; use a location outside the repository for custom outputs.

## Files

| File | Role |
|---|---|
| main.py | Experiment entry point and task loop |
| loaddataset.py | Data loading, preprocessing, and protocols |
| params.py | Parameters and model selection |
| model.py | Networks, initialization, GRL, and spectral projection |
| train.py | Losses and optimization |
| evaluate.py | Accuracy evaluation and result summaries |
| utils.py | Sampling and saving |
