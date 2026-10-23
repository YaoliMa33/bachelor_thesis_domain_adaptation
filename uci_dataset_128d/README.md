# UCI Gas Sensor Features

[Project overview](../README.md) · [Training guide](../uci_training/README.md)

[Gas Sensor Array Drift at Different Concentrations (UCI 270)](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) contains 13,910 measurements of six gases across ten batches. Each measurement has 128 features: eight descriptors from each of 16 sensors.

## Obtain the data

Provider-owned batch files are not redistributed in this source release. After reviewing the provider terms, run from the repository root:

```bash
python uci_dataset_128d/download.py
```

The script downloads the official archive and verifies all ten batch files against [checksums.json](checksums.json) before installing them into `uci_dataset_128d/data/`. This local data directory is ignored by Git and excluded by the release builder. Existing differing files are never overwritten. To use an already downloaded official ZIP without a network connection:

```bash
python uci_dataset_128d/download.py --archive PATH_TO_OFFICIAL_ZIP
```

Training and MSPA evaluation use `uci_dataset_128d/data/` by default. `--data PATH` can still point to another directory containing all ten files. Obtaining a local copy does not change the provider's terms.

## Format and labels

The locally downloaded `batch1.dat` through `batch10.dat` retain the official file contents:

```text
label; concentration 1:value ... 128:value
```

Labels and concentration are excluded from model inputs. The loader subtracts one from each numeric label and applies asinh followed by source-fitted z-score standardization, without changing the files.

| File label | Training index | Gas |
|---:|---:|---|
| 1 | 0 | Ethanol |
| 2 | 1 | Ethylene |
| 3 | 2 | Ammonia |
| 4 | 3 | Acetaldehyde |
| 5 | 4 | Acetone |
| 6 | 5 | Toluene |

[gas_labels.json](gas_labels.json) follows the source page’s explicit numeric-label description. Its historical class-count table uses a conflicting order; the data have not been relabeled to match that table.

## Sources and license

Dataset citation: Vergara, A. (2012). *Gas Sensor Array Drift at Different Concentrations*. UCI Machine Learning Repository. [DOI: 10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M).

The source page also requests citation of both papers:

- Vergara et al. (2012). *Chemical gas sensor drift compensation using classifier ensembles*. Sensors and Actuators B: Chemical.
- Rodriguez-Lujan et al. (2014). *On the calibration of sensor arrays for pattern recognition using the minimal number of experiments*. Chemometrics and Intelligent Laboratory Systems, 130.

As checked on 23 October 2026, the UCI License section lists CC BY 4.0, while the older Additional Information text restricts use to research and excludes commercial use. This repository records both statements without resolving the conflict or granting further rights. This source release avoids redistributing the dataset; it supplies the official download link and byte-verification hashes only. Confirm the applicable terms with the provider before unrestricted redistribution or commercial use. The project’s code and ST data licenses do not replace the UCI terms.
