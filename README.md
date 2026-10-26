# Gas Sensor Drift Compensation

Domain adaptation for gas recognition on the self-collected ST and public UCI datasets: DANN, CDAN, latent class-center alignment (LC), entropy conditioning (+E), and multi-seed probability averaging (MSPA).

## Bachelor thesis

[**Compensating Gas Sensor Drift via Domain Adaptation for Robust Gas Discrimination**](thesis/Compensating_Gas_Sensor_Drift_via_Domain_Adaptation_for_Robust_Gas_Discrimination.pdf)

- **Author:** Yaoli Ma
- **Advisor:** Dr. Fan Han
- **Supervisor:** Prof. Dr. Achim J. Lilienthal
- **Chair:** Chair of Perception for Intelligent Systems
- **University:** Technical University of Munich


## Repository guide

| Folder | Contents |
|---|---|
| [st_dataset](st_dataset/README.md) | 54-dimensional training inputs, labels, and license |
| [st_training](st_training/README.md) | ST protocols, parameters, and evaluation |
| [uci_dataset_128d](uci_dataset_128d/README.md) | Official UCI download, checksums, labels, and citations |
| [uci_training](uci_training/README.md) | UCI protocols, +E, and MSPA |

## Environment

Verified on 64-bit Windows with Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.0, and PyTorch 2.14.0+cpu. Training uses one CPU thread. With Python 3.12 installed:

```bash
git clone https://github.com/YaoliMa33/bachelor_thesis_domain_adaptation.git
cd bachelor_thesis_domain_adaptation
python -m venv .venv
```

Activate with `.venv\Scripts\activate.bat` in Windows Command Prompt, `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a Unix shell. For Windows/Linux, install the CPU build and pinned dependencies:

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On macOS, use only `python -m pip install -r requirements.txt` with the default package index. The verified environment above is Windows.

## Reproduce the experiments

Run all commands from the repository root. ST inputs are included; download UCI inputs once:

```bash
python uci_dataset_128d/download.py
```

Start with two seeds and the full training budget:

```bash
python st_training/dann_cdan_lc/main.py --seeds 0 1
python st_training/dann_cdan_lc/evaluate.py --seeds 0 1
python uci_training/dann_cdan_lc/main.py --seeds 0 1
python uci_training/dann_cdan_lc/evaluate.py --seeds 0 1
```

Omit `--seeds 0 1` for seeds 0–19. Defaults are four models, 300 updates for ST and 450 for UCI. The training guides explain custom parameters, +E, and MSPA.

Reproduction starts from the supplied features. Raw acquisition and feature-extraction tools are outside this repository's scope. Results and checkpoints are generated locally and are not bundled. Records include data/code hashes and runtime versions; evaluation checks the training inputs and reports both equal-task and sample-weighted accuracy for ST and UCI base/+E experiments.

## Licenses

Code and general documentation use [MIT](LICENSE); ST features use [CC BY 4.0](st_dataset/LICENSE.md). [UCI data](uci_dataset_128d/README.md#sources-and-license) are obtained from their provider under its terms. These licenses do not apply to the thesis PDF.

## References

- Ganin et al. (2016). [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html).
- Long et al. (2018). [Conditional Adversarial Domain Adaptation](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html).
