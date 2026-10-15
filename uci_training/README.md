# UCI 训练

[项目首页与环境安装](../README.md) · [UCI 数据说明](../uci_dataset_128d/README.md)

| 模块 | 内容 |
|---|---|
| [dann_cdan_lc](dann_cdan_lc) | DANN、DANNLC、CDAN、CDANLC 的共用实现 |
| [entropy/train.py](entropy/train.py) | 熵条件加权（+E），复用基础训练循环 |
| [mspa](mspa/README.md) | 多种子训练入口和 softmax 概率平均评估 |

## 基础模型

从仓库根目录执行，默认四个模型、种子 0–19：

```bash
python uci_training/dann_cdan_lc/main.py
python uci_training/dann_cdan_lc/test.py
```

`--uci-protocol long|medium|both` 选择协议，默认 `both`：

- Setting B（长期）：B1 → B2,…,B10，共九个任务。
- Setting C（中期）：Bk → B(k+1)，k=1,…,9，包含 B5 → B6。

每个任务重新初始化，标准化按本任务源批次拟合。运行 `both` 时，B1 → B2 只训练一次，计入两个协议。

`--data PATH` 指定含十个 batch 文件的目录；`--output PATH` 指定输出根目录；`--seeds 0 1` 可选择种子。训练与汇总的输出路径、种子及协议须一致。修改 [params.py](dann_cdan_lc/params.py) 中的 `METHODS` 可选择模型，例如 `("CDANLC",)`。

## 共用参数与方法

| 参数 | 默认值 |
|---|---|
| 输入／类别／特征隐藏维度 | 128／6／200 |
| 特征提取器 | Linear → BatchNorm → ReLU；源、目标联合 BN |
| 域判别器隐藏层／谱范数上限 | 64 → 64／1.0 |
| 每域 minibatch／更新次数 | 16／450（45 × 10） |
| 优化器／初始学习率 | SGD，momentum=0.9／0.03；weight decay=0 |
| 学习率调度 | 每记录区间余弦衰减，eta_min=1e-6 |
| 域损失／LC 权重 | 0.8／0.2；DANN、CDAN 的 LC 权重为 0 |
| GRL | gamma=10，覆盖全部 450 步 |
| 源域采样 | 按类别比例分配并分层循环采样 |
| 目标域采样 | 跨记录区间连续采样，拼接余样本 |

`epoch` 是固定 10 次更新的记录区间。预处理为 asinh 后按源域统计量做 z-score。初始化在网络构建后按种子重置。DANN 使用特征作域输入，CDAN 使用特征与 detached 类别概率的外积。LC 对齐源类中心和目标软概率加权中心，只计入源小批次出现的类别。域判别器在初始化及每次更新后进行谱范数投影。

基础模块分工：`main.py` 管理任务；`loaddataset.py` 读取和预处理数据；`model.py` 定义网络；`train.py` 计算损失及优化；`test.py` 评估和汇总；`utils.py` 处理采样及保存。

## 熵条件加权（+E）

支持四个基础模型的 +E 变体，默认仅长期协议，其他参数沿用共用配置：

```bash
python uci_training/entropy/train.py
python uci_training/dann_cdan_lc/test.py --output uci_training/entropy/results --uci-protocol long
```

使用 `H = -sum(p * log(p + 1e-5))`、`w = 1 + exp(-H)` 加权域分类损失，源、目标权重分别归一化；归一化分母 detach，熵分支使用相同 GRL 系数。参考 [CDAN 官方实现](https://github.com/thuml/CDAN/blob/master/pytorch/loss.py)，DANN 系列的 +E 为本项目扩展。

默认输出至 `entropy/results/UCI/`。JSON 保留基础模型名，并以 `entropy_conditioning`、`variant` 标明 +E；汇总中的 LC 增益比较同为 +E 的模型，不表示 E 相对基础模型的增益。

## 多随机种子概率平均（MSPA）

```bash
python uci_training/mspa/train.py
python uci_training/mspa/evaluate.py
```

训练直接复用基础模型，不启用 +E。评估需要所选模型与任务的完整种子池 0–19；默认每组 7 个成员，先平均 softmax 概率，再取预测类别。已有基础权重时用 `--members uci_training/dann_cdan_lc/results` 复用。

成员分组、数量分析、输出与重叠组标准差的解释见 [MSPA 说明](mspa/README.md)。

## 输出与统计

基础训练输出至 `dann_cdan_lc/results/UCI/`，按种子、任务、模型保存 JSON 和最终 `.pt` 权重；相同路径重跑会覆盖同名文件。评估固定最后一步模型。`test.py` 汇总已有结果，不重新训练。

每个种子先对九个任务准确率等权平均，再报告种子间均值和样本标准差（ddof=1）。JSON 保存逐种子均值、均值／SD 和配对 LC 增益，CSV 保存均值和平均增益。一个种子的 SD 为 `null`；仅当同系列两个模型都被选中时计算 LC 增益。

目标标签不参与反向传播或 checkpoint 选择；参数与预算曾参考目标评估结果选择，结果属于探索性比较。默认结果、权重与缓存已被 `.gitignore` 排除，自定义输出建议放在仓库外。
