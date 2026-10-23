# Gas Sensor Drift Compensation

Domain adaptation experiments for gas recognition using the self-collected ST dataset and the public UCI dataset. The repository includes DANN, CDAN, their latent class-center alignment (LC) variants, and UCI extensions for entropy conditioning (+E) and multi-seed probability averaging (MSPA).

## Repository guide

| Folder | Contents |
|---|---|
| [st_dataset](st_dataset/README.md) | 54-dimensional training inputs, labels, and license |
| [st_training](st_training/README.md) | ST training protocols, parameters, and evaluation |
| [uci_dataset_128d](uci_dataset_128d/README.md) | Official UCI download, checksums, labels, and citations |
| [uci_training](uci_training/README.md) | Long- and medium-term UCI experiments, +E, and MSPA |

## Environment

Verified on 64-bit Windows with Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.0, and PyTorch 2.14.0+cpu. Training uses one CPU thread. Start with Python 3.12 and a separate environment:

```bash
git clone https://github.com/YaoliMa33/bachelor_thesis_domain_adaptation.git
cd bachelor_thesis_domain_adaptation
python -m venv .venv
```

Activate it with `.venv\Scripts\activate.bat` in Windows Command Prompt, `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a Unix shell. Then install the dependencies (the CPU-wheel command below is for Windows/Linux):

```bash
python -m pip install numpy==2.5.3 scikit-learn==1.9.0
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Run the commands in each training guide from the repository root. Defaults are four models and seeds 0–19, with 300 updates for ST and 450 for UCI. Results and model checkpoints are generated locally and are not included.

## Reproduce the experiments

Run from the repository root. The ST feature table is included. Download the UCI inputs once before UCI training:

```bash
python uci_dataset_128d/download.py
```

For an initial run using two seeds and the full training budget:

```bash
python st_training/dann_cdan_lc/main.py --seeds 0 1
python st_training/dann_cdan_lc/test.py --seeds 0 1
python uci_training/dann_cdan_lc/main.py --seeds 0 1
python uci_training/dann_cdan_lc/test.py --seeds 0 1
```

Omit `--seeds 0 1` to use all 20 seeds. The [ST guide](st_training/README.md) and [UCI guide](uci_training/README.md) describe the protocols, parameters, +E and MSPA. Training and evaluation save results locally; no precomputed results, checkpoints or experiment reports are included. This repository supports reproduction from the supplied feature inputs; raw acquisition and feature-extraction tools are outside its scope.

## Licenses

The project's Python code and general documentation use the [MIT license](LICENSE). The ST feature data use [CC BY 4.0](st_dataset/LICENSE.md). [UCI data](uci_dataset_128d/README.md#sources-and-license) are downloaded directly from their provider and are not redistributed in this repository.

## References

- Ganin et al. (2016). [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html).
- Long et al. (2018). [Conditional Adversarial Domain Adaptation](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html).
