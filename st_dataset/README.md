# ST Dataset

[Project overview](../README.md) · [Training guide](../st_training/README.md)

The ST dataset contains 303 recordings from three gas classes across six batches.

| File or folder | Contents |
|---|---|
| [st_54d/features_54d.csv](st_54d/features_54d.csv) | Extracted features used directly for training |
| [st_raw/st_raw_csv.zip](st_raw/st_raw_csv.zip) | The 303 original time-series CSV files |
| [st_raw/sample_manifest.csv](st_raw/sample_manifest.csv) | Sample IDs, labels, batches, and raw file paths |
| [extract_features.py](extract_features.py) | Rebuild and verify all 54 features from the original ZIP |
| [FEATURE_EXTRACTION.md](FEATURE_EXTRACTION.md) | Exact windows, formulas and cleaning rules |
| [arduino](arduino/README.md) | Hardware provenance and a standalone serial-command parser |

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

## Raw recordings and license

Training uses the preserved feature table directly. To reconstruct and verify it, run `python st_dataset/extract_features.py --verify --clean-archive` from the repository root. See the [extraction specification](FEATURE_EXTRACTION.md) for windows and formulas, and the [raw data guide](st_raw/README.md) for timestamps and documented cleaning.

The recordings, manifest, and extracted features use [CC BY 4.0](LICENSE.md). The original hardware firmware is excluded from this source distribution; the included parser has [separate provenance notes](arduino/README.md#provenance-and-license).
