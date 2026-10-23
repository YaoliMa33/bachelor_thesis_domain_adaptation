# ST Training Data

[Project overview](../README.md) · [Training guide](../st_training/README.md)

[st_54d/features_54d.csv](st_54d/features_54d.csv) contains the input features for 303 samples from three gas classes across six batches. Training reads this file directly; no hardware or raw-data processing is required.

## Features and labels

Each of six sensor channels contributes nine features, giving 54 inputs. The loader reads `s1_*` through `s6_*` in file order. The descriptors are `exp_mean`, `exp_absmax`, `exp_span`, `exp_tail`, `exp_auc_abs`, `rec_mean`, `rec_tail`, `rec_slope`, and `overall_std`. Nine additional metadata columns bring the CSV total to 63; these columns are not model inputs.

Class indices are `air → 0`, `alcohol → 1`, and `acetone → 2`.

| Batch | Value in the batch column | Samples |
|---|---|---:|
| B1 | source | 30 |
| B2 | batch1 | 28 |
| B3 | batch2 | 84 |
| B4 | batch3 | 81 |
| B5 | batch4 | 46 |
| B6 | batch5 | 34 |

The features were extracted after per-record baseline correction. They have not been transformed with asinh or z-score standardization. The training loader applies both in memory, fitting the standardization statistics on the source domain only.

## Run and license

From the repository root:

```bash
python st_training/dann_cdan_lc/main.py
python st_training/dann_cdan_lc/test.py
```

Use `--data PATH` to supply another feature CSV in the same format. The supplied ST feature data and dataset documentation use [CC BY 4.0](LICENSE.md). This directory contains training inputs, not model outputs or experiment reports.
