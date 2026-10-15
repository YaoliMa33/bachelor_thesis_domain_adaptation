# Gas Sensor Drift Compensation

基于 ST 自采数据和 UCI 公开数据的气体传感器漂移补偿实验，包含 DANN、CDAN 及其 LC 扩展，以及 UCI 熵条件加权（+E）和多随机种子概率平均（MSPA）。

## 项目目录

| 文件夹 | 内容与使用说明 |
|---|---|
| [st_dataset](st_dataset/README.md) | ST 原始采集 CSV、54 维特征、标签与批次 |
| [st_training](st_training/README.md) | ST 三种协议的训练、参数与统计汇总 |
| [uci_dataset_128d](uci_dataset_128d/README.md) | UCI 128 维特征、标签、数据来源与引用 |
| [uci_training](uci_training/README.md) | UCI 长期／中期训练、+E 和 MSPA |

## 环境

已验证：Python 3.12.14、NumPy 2.5.3、scikit-learn 1.9.0、PyTorch 2.14.0+cpu。训练使用 CPU 单线程。

```bash
python -m pip install numpy==2.5.3 scikit-learn==1.9.0
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

运行命令见各训练文件夹的 README，均从仓库根目录执行。默认运行四个模型、种子 0–19；ST 为 300 次更新，UCI 为 450 次更新。仓库提供代码和数据，不包含实验结果或模型权重。

## 许可

代码与通用文档采用 [MIT](LICENSE)，自采 ST 数据采用 [CC BY 4.0](st_dataset/LICENSE.md)。UCI 数据保留其[原始来源、引用及授权说明](uci_dataset_128d/README.md#来源与许可)。

## 方法参考

- Ganin et al. (2016). [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html).
- Long et al. (2018). [Conditional Adversarial Domain Adaptation](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html).
