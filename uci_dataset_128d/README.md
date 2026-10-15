# UCI Gas Sensor Array Drift at Different Concentrations

Source: [UCI dataset 270](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations).

The ten batch files contain 13,910 measurements with 128 features each. The loader preserves numeric labels and subtracts one for zero-based class indices; it does not use concentration as an input. `gas_labels.json` follows the source page's explicit numeric-label description. The historical class-count table and numeric-label description differ in their ordering; the files have not been relabeled using that table.

## Attribution

Dataset: Vergara, A. (2012). *Gas Sensor Array Drift at Different Concentrations*. UCI Machine Learning Repository. [DOI: 10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M).

The source page requests citation of both papers:

- Vergara et al. (2012). *Chemical gas sensor drift compensation using classifier ensembles*. Sensors and Actuators B: Chemical.
- Rodriguez-Lujan et al. (2014). *On the calibration of sensor arrays for pattern recognition using the minimal number of experiments*. Chemometrics and Intelligent Laboratory Systems, 130.

## Source licensing statements

Checked on 16 October 2026: the UCI page's License section identifies CC BY 4.0, while its older Additional Information text still limits use to research and excludes commercial use. These statements conflict. This repository records both statements and does not resolve that conflict or grant additional rights. Confirm the applicable terms with the dataset provider before treating the files as unrestricted redistribution or commercial-use material. Any eventual license for this project's own code or ST data does not replace the UCI data terms.
