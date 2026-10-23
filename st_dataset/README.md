# ST Training Data

[Project overview](../README.md) · [Training guide](../st_training/README.md)

[st_54d/features_54d.csv](st_54d/features_54d.csv) supplies 303 samples from three gas classes across six batches. Training reads it directly; no hardware or raw-data processing is required.

## Features and labels

Six sensor channels contribute nine features each, giving 54 inputs. The loader reads `s1_*` through `s6_*` in file order. Each channel provides `exp_mean`, `exp_absmax`, `exp_span`, `exp_tail`, `exp_auc_abs`, `rec_mean`, `rec_tail`, `rec_slope`, and `overall_std`. The other nine columns are metadata, excluded from model inputs.

Class indices: `air → 0`, `alcohol → 1`, `acetone → 2`.

| Batch | Value in the batch column | Samples |
|---|---|---:|
| B1 | source | 30 |
| B2 | batch1 | 28 |
| B3 | batch2 | 84 |
| B4 | batch3 | 81 |
| B5 | batch4 | 46 |
| B6 | batch5 | 34 |

Features were extracted after per-record baseline correction. The loader applies asinh and z-score standardization in memory, fitting statistics on the source domain only; neither transformation has already been applied to the stored table.

The `source_kind` column records provenance. All six source-batch air samples are `baseline_as_air`; target air samples include `new` and `reference_as_air`. This column is not a model input.

## Usage and license

Follow the [training commands](../st_training/README.md#run). The default input is the included feature table; `--data PATH` accepts another CSV with the same schema. ST features and dataset documentation use [CC BY 4.0](LICENSE.md).
