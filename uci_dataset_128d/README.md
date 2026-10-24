# UCI Gas Sensor Features

[Project overview](../README.md) · [Training guide](../uci_training/README.md)

[Gas Sensor Array Drift at Different Concentrations (UCI 270)](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) contains 13,910 measurements of six gases across ten batches. Each measurement has 128 features: eight descriptors from each of 16 sensors.

## Obtain the data

UCI batch files are not bundled. Review the provider terms, then run from the repository root:

```bash
python uci_dataset_128d/download.py
```

The script downloads the official archive, verifies all ten files against [checksums.json](checksums.json), and installs them into the Git-ignored `uci_dataset_128d/data/`. Existing differing files are never overwritten. For an already downloaded official ZIP:

```bash
python uci_dataset_128d/download.py --archive PATH_TO_OFFICIAL_ZIP
```

Training, summarization and MSPA also verify hashes in the default directory. `--data PATH` accepts another directory containing ten valid batch files; their actual hashes are recorded and checked when results are reused.

## Format and labels

The official `batch1.dat` through `batch10.dat` files use:

```text
label; concentration 1:value ... 128:value
```

Labels and concentration are excluded from inputs. The loader subtracts one from each label, then applies asinh and source-fitted z-score standardization to the features in memory.

| File label | Training index | Gas |
|---:|---:|---|
| 1 | 0 | Ethanol |
| 2 | 1 | Ethylene |
| 3 | 2 | Ammonia |
| 4 | 3 | Acetaldehyde |
| 5 | 4 | Acetone |
| 6 | 5 | Toluene |

[gas_labels.json](gas_labels.json) follows the provider's explicit numeric-label description, rather than its historical class-count table's conflicting order. Files are not relabeled.

## Sources and license

Vergara, A. (2012). *Gas Sensor Array Drift at Different Concentrations*. UCI Machine Learning Repository. [DOI: 10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M).

The provider also requests these references:

- Vergara et al. (2012). *Chemical gas sensor drift compensation using classifier ensembles*. Sensors and Actuators B: Chemical.
- Rodriguez-Lujan et al. (2014). *On the calibration of sensor arrays for pattern recognition using the minimal number of experiments*. Chemometrics and Intelligent Laboratory Systems, 130.

As checked on 23 October 2026, the UCI License section lists CC BY 4.0, while older Additional Information restricts use to research and excludes commercial use. This repository does not resolve that conflict or redistribute the data. Confirm the applicable terms with the provider before unrestricted redistribution or commercial use; the project licenses do not replace those terms.
