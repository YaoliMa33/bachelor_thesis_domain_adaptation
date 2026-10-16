# UCI Gas Sensor Features

[Project overview](../README.md) · [Training guide](../uci_training/README.md)

[Gas Sensor Array Drift at Different Concentrations (UCI 270)](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) contains 13,910 measurements of six gases across ten batches. Each measurement has 128 features: eight descriptors from each of 16 sensors.

## Format and labels

`batch1.dat` through `batch10.dat` retain the official file contents:

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

As of 16 October 2026, the UCI License section lists CC BY 4.0, while the older Additional Information text restricts use to research and excludes commercial use. This repository records both statements without resolving the conflict or granting further rights. Confirm the applicable terms with the provider before unrestricted redistribution or commercial use. The project’s code and ST data licenses do not replace the UCI terms.
