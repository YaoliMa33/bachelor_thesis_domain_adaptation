# Gas Sensor Drift: ST 与 UCI 实验总览

本目录保存 ST、UCI 特征数据和 DANN、DANNLC、CDAN、CDANLC 多文件训练代码，以及独立的熵条件加权（+E）和多随机种子概率平均（MSPA）入口。ST 使用 300 步；UCI 长期和中期共用 450 步及同一套参数。

## 目录

```text
GitHub/
├── README.md                         # 本总览
├── LICENSE                           # 代码与通用文档：MIT
├── .gitignore                        # 排除本地生成结果与缓存
├── st_training/dann_cdan_lc/          # ST 专用训练代码
├── st_dataset/st_54d/features_54d.csv # ST 提取后的 54 维特征
├── st_dataset/st_raw/                # 原始时序CSV压缩包、样本清单与说明
├── uci_training/dann_cdan_lc/         # UCI 原模型及共用模块
├── uci_training/entropy/train.py      # UCI 熵条件加权（+E）训练入口
├── uci_training/mspa/                 # MSPA多种子训练入口与概率平均评估
└── uci_dataset_128d/
    ├── batch1.dat ... batch10.dat    # UCI 官方 128 维特征
    ├── gas_labels.json              # 气体标签对应关系
    └── README.md                    # UCI 来源、引用与授权说明
```

两个训练目录各有 `main.py`、`loaddataset.py`、`model.py`、`params.py`、`train.py`、`test.py`、`utils.py`。基础训练代码包含上述四个域对抗模型。`entropy` 与 `mspa` 复用 UCI 公共模块；训练结果、模型权重和其他对比方法的结果不随本目录发布。

| 文件 | 职责 |
|---|---|
| main.py | 实验入口、任务循环、训练与评估调用 |
| loaddataset.py | 数据读取、任务定义、训练时的源域标准化 |
| model.py | 网络、初始化、梯度反转和域判别器谱范数投影 |
| params.py | 参数和 METHODS 模型选择 |
| train.py | 分类、域对抗、LC 损失与优化 |
| test.py | 模型准确率评估、已保存结果的统计汇总 |
| utils.py | 批次采样及模型、结果保存 |

## 环境与运行

核验环境：Python 3.12.14、NumPy 2.5.3、scikit-learn 1.9.0、PyTorch 2.14.0+cpu。代码使用 CPU 和单个 PyTorch 线程。依赖在此统一说明，不另设 requirements 文件。

```bash
python -m pip install numpy==2.5.3 scikit-learn==1.9.0
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

以下命令从 `GitHub` 根目录执行：

```bash
# ST：三个设置，四个模型，默认种子 0–19
python st_training/dann_cdan_lc/main.py
python st_training/dann_cdan_lc/test.py

# UCI：长期和中期一起运行，默认种子 0–19
python uci_training/dann_cdan_lc/main.py
python uci_training/dann_cdan_lc/test.py

# UCI：仅中期
python uci_training/dann_cdan_lc/main.py --uci-protocol medium
python uci_training/dann_cdan_lc/test.py --uci-protocol medium
```

训练入口自动按脚本位置定位对应数据目录；可用 `--data PATH` 指定其他位置。结果默认保存在各训练目录的 `results/ST/` 或 `results/UCI/`。`--output PATH` 可修改输出根目录；训练与汇总应使用相同的 `--output`、`--seeds` 和 UCI 协议参数。每次任务保存 JSON 结果和最终模型 `.pt`；同名输出会被覆盖。`test.py` 单独执行时汇总已有结果，不重新训练。`summary.json` 包含每个种子的等权均值、总体均值、样本标准差和配对增益；`summary.csv` 提供均值与平均增益。只汇总一个种子时，样本标准差未定义，JSON 中记为 `null`。

## UCI 熵条件加权实验（+E）

`uci_training/entropy/train.py` 保存 DANN+E、DANNLC+E、CDAN+E、CDANLC+E 的训练实现，可直接运行。它复用相邻 `dann_cdan_lc` 下的 `main.py`、`model.py`、`params.py`、`loaddataset.py`、`test.py`、`train.py` 和 `utils.py`，并直接调用公共训练循环和 LC 损失，不复制基础优化过程。模型选择及公共参数仍在 `uci_training/dann_cdan_lc/params.py` 中设置。

与原 `train.py` 相比，核心改动是新增 `entropy_domain_loss()`，并在 `losses()` 中用它替换普通域对抗损失。根据分类概率计算 `H = -sum(p * log(p + 1e-5))` 和 `w = 1 + exp(-H)`，分别归一化源域与目标域权重，再加权每个样本的域分类损失。归一化分母 detach，熵分支使用原有 GRL 系数反转梯度。CDAN 的条件输入和 LC 的类别中心计算保持不变。实现参考 [CDAN 官方 loss.py](https://github.com/thuml/CDAN/blob/master/pytorch/loss.py)；DANN 系列的 +E 是本实验的扩展。

从 `GitHub` 根目录运行：

```bash
# 默认：Setting B 长期漂移，B1→B2…B10，四个模型，种子 0–19
python uci_training/entropy/train.py

# 显式指定协议与种子，例如先运行两个种子
python uci_training/entropy/train.py --uci-protocol long --seeds 0 1

# 复用原有汇总程序，必须指向 E 的结果目录
python uci_training/dann_cdan_lc/test.py --output uci_training/entropy/results --uci-protocol long --seeds 0 1
```

E 默认输出到 `uci_training/entropy/results/UCI/`，与原模型结果分开。`--data`、`--output`、`--seeds`、`--uci-protocol` 沿用原入口；汇总时 seeds 必须与已有结果一致，计算样本标准差至少需要两个种子。默认参数为 45×10 次更新、`lambda_d=0.8`、LC 模型 `lambda_lc=0.2`。

为兼容原汇总程序，文件名和 `method` 字段仍使用 `DANN`、`DANNLC`、`CDAN`、`CDANLC`；每份训练 JSON 额外保存 `entropy_conditioning: true` 与带 `+E` 的 `variant` 字段。该目录下的汇总结果应按 +E 模型解读；汇总中的 `paired_LC_gains` 比较同为 +E 的 LC/非 LC 模型，并不是 E 相对原模型的增益。后者需要将两个结果目录按相同种子、任务配对比较。

## UCI 多随机种子概率平均（MSPA）

`uci_training/mspa/train.py` 直接调用 `dann_cdan_lc` 的原始多种子训练入口，只将默认输出改为 `mspa/results`。DANN、CDAN、LC的损失、网络、预处理与采样仍由 `dann_cdan_lc` 定义；`mspa` 不复制基础训练实现，也不默认启用 +E。

`uci_training/mspa/evaluate.py` 加载种子0–19的最终模型，同一方法内平均softmax概率，再取最大概率类别。默认每组7个成员，采用论文中的20个循环分组。每组内先对九个任务等权平均，再报告组间均值和样本标准差。组间共享成员，标准差表示固定种子池内的组合波动。20成员只形成一个唯一组合，不报告组间标准差。

```bash
python uci_training/mspa/train.py
python uci_training/mspa/evaluate.py
# 已有原模型的完整20种子权重时，可直接复用，无需重复训练：
python uci_training/mspa/evaluate.py --members uci_training/dann_cdan_lc/results
```

训练与评估默认覆盖长期和中期；可用 `--uci-protocol long` 或 `medium` 选择。默认中期保留全部九个任务，包括B5→B6。成员数量分析、文件格式、分组偏移与输出位置见 [MSPA说明](uci_training/mspa/README.md)。评估不会重新训练，也不会按目标准确率挑选成员。

## 数据与标签

**特征文件保留提取后的数值，没有额外施加 asinh、z-score、缩放、行重排或标签重标；ST 原始时序另存于 st_raw。** 训练时才在内存中执行 `asinh → 当前源域拟合的 z-score`；训练不修改数据文件。

### ST

- 303 条样本、54 个特征列：6 路传感通道各 9 个描述量。
- CSV 另有 9 个元数据列，包含标签和批次，不作为模型输入。因此 CSV 总列数为 63，输入维度仍为 54。
- 特征列按原文件顺序读取 `s1_*` 至 `s6_*`。每个传感器的描述量为 `exp_mean`、`exp_absmax`、`exp_span`、`exp_tail`、`exp_auc_abs`、`rec_mean`、`rec_tail`、`rec_slope`、`overall_std`。
- 训练标签：`air→0`、`alcohol→1`、`acetone→2`，对应论文 Air、Alcohol、Acetone。
- 批次：`source→B1`、`batch1→B2`，依次到 `batch5→B6`；样本数依次为 30、28、84、81、46、34。
- `st_54d/features_54d.csv` 是论文使用的已提取特征。`st_raw/st_raw_csv.zip` 保存对应的303个原始采集CSV，按 air/alcohol/acetone 分类；`st_raw/sample_manifest.csv` 提供样本、批次与文件对应关系。详见 [原始数据说明](st_dataset/st_raw/README.md)。`target_file` 是相对于解压目录的路径，训练不需要读取这些原始时序文件。

### UCI

- 13,910 条样本、10 个批次，每条样本 128 个官方特征（16 个传感器 × 8 个描述量）。
- 文件格式：`label; concentration 1:value ... 128:value`。标签、浓度均不属于 128 个特征；模型不使用浓度。
- 文件与官网下载压缩包中的 `batch1.dat` 至 `batch10.dat` 逐字节一致。

| 官方数字标签 | 训练索引 | 气体 |
|---:|---:|---|
| 1 | 0 | Ethanol／乙醇 |
| 2 | 1 | Ethylene／乙烯 |
| 3 | 2 | Ammonia／氨 |
| 4 | 3 | Acetaldehyde／乙醛 |
| 5 | 4 | Acetone／丙酮 |
| 6 | 5 | Toluene／甲苯 |

采用项目已确认的官网数字标签说明。官网历史样本数量表的排列与数字标签说明存在冲突；本项目不据数量表重标原始标签。数据来源：[UCI Dataset 270](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)；引用：Vergara, A. (2012), *Gas Sensor Array Drift at Different Concentrations*, UCI Machine Learning Repository, [DOI: 10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M)。数据来源、两篇要求引用的研究论文及官网的授权说明见 [UCI 数据说明](uci_dataset_128d/README.md)。

## 共同方法与参数差异

四个模型共用 Linear→BatchNorm→ReLU 特征提取器（隐藏维度 200）及线性分类头，源域和目标域一起通过 BatchNorm。DANN 用特征作域输入；CDAN 用特征与已 detach 的类别概率外积。LC 使用源类中心和目标软概率加权中心，仅计入源小批次出现的类别。域判别器线性权重在初始化及每次更新后作谱范数投影。ST 源域使用随机小批次；UCI 源域使用分层循环采样。评估固定最后一步模型。

| 参数 | ST | UCI：长期、中期共用 |
|---|---:|---:|
| 输入维度／类别数 | 54／3 | 128／6 |
| 特征隐藏维度 | 200 | 200 |
| 域判别器隐藏层 | 32→32 | 64→64 |
| 每域 batch size | 20 | 16 |
| 更新次数 | 300（60×5） | 450（45×10） |
| 优化器 | Adam | SGD，momentum 0.9 |
| 初始学习率 | 0.005 | 0.03 |
| Weight decay | 0.0001 | 0 |
| 域损失权重 | 0.9 | 0.8 |
| LC 权重，仅 LC 模型 | 1.2 | 0.2 |
| 学习率调度 | 固定 | 逐记录区间余弦衰减，eta_min=1e-6 |
| GRL gamma | 3 | 10 |
| GRL 升温 | 前150步递增，后150步固定 | 全450步递增 |
| 域判别器谱范数上限 | 0.75 | 1.0 |
| 源域采样 | 随机加载器，无固定类别配额 | 按源类别比例分配并循环采样 |
| 目标域采样 | 每记录区间重启加载器，丢弃不足批次 | 跨记录区间连续采样，余样本拼接 |

当前 ST 的 `source_sampling="loader"`，不固定每批三类的 4/8/8 配额。因此，当前默认设置不等同于此前使用分层源域采样的论文实验。

代码中的 epoch 是固定更新次数的记录区间，不表示完整遍历一次数据。ST 在全部300步中保持优化器状态连续。UCI 长期与中期、四模型的公共训练参数一致。

修改各目录 `params.py` 顶部的 `METHODS` 可选择模型：

```python
METHODS = ("DANN", "DANNLC", "CDAN", "CDANLC")
```

`DANN`、`CDAN` 关闭 LC（权重0），`DANNLC`、`CDANLC` 开启 LC。单模型元组须保留逗号，例如 `("CDANLC",)`。汇总支持所选模型子集，仅在同系列的基线和 LC 模型同时被选中时计算配对 LC 增益。汇总时的 METHODS、种子与协议应与训练保持一致。

## 实验协议与统计

- ST Setting A / Protocol 1（代码记为 Setting1）：B1 为有标签源域，B2–B6 合并为无标签目标池；分别评价每个目标批次。
- ST Setting A / Protocol 2（代码记为 Setting2）：独立训练 B1→Bk，k=2,…,6。
- ST Setting A / Protocol 3（代码记为 Setting3）：每个任务从头训练，目标池为 B2,…,Bk，评价 Bk；不继承上一任务模型。
- UCI 长期：独立训练 B1→Bk，k=2,…,10。
- UCI 中期：独立训练 Bk→B(k+1)，k=1,…,9。标准化按各任务源批次拟合。
- 每个种子内任务等权平均，再计算种子间均值和样本标准差（ddof=1）。目标标签不参与反向传播或末步 checkpoint 选择；参数和预算曾参考目标评估结果选择，因此结果属于探索性比较。

## 方法参考

- Ganin et al. (2016), [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html).
- Long et al. (2018), [Conditional Adversarial Domain Adaptation](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html).

## 发布状态与授权

- 原创训练、评估代码与通用文档：[MIT License](LICENSE)。
- 自采 ST 原始数据、样本清单、54 维特征及其数据文档：[CC BY 4.0](st_dataset/LICENSE.md)。使用或改编时注明来源、许可及修改情况。
- UCI 数据：保留其第三方来源与授权，不适用本项目 MIT 或 ST 数据许可。官网同时存在 CC BY 4.0 与旧版研究用途限制，详见 [UCI 数据说明](uci_dataset_128d/README.md)。

默认生成的 results 目录、模型权重及 Python 缓存由根目录 .gitignore 排除。使用自定义输出路径时，请将输出放在仓库之外。本目录不包含训练结果、检查报告或历史实验归档。
