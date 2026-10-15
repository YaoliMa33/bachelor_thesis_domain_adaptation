# ST 训练

[项目首页与环境安装](../README.md) · [ST 数据说明](../st_dataset/README.md)

`dann_cdan_lc` 实现 DANN、DANNLC、CDAN、CDANLC，覆盖 Setting A 的三个目标数据访问协议。

## 运行

以下命令从仓库根目录执行，默认训练四个模型、种子 0–19：

```bash
python st_training/dann_cdan_lc/main.py
python st_training/dann_cdan_lc/test.py
```

`--data PATH` 指定特征 CSV；`--output PATH` 指定输出根目录；`--seeds 0 1` 可选择种子。训练与汇总使用相同的输出路径和种子。模型选择及参数在 [params.py](dann_cdan_lc/params.py) 中设置；单模型示例为 `METHODS = ("CDANLC",)`。

## 协议与参数

B1 始终为有标签源域，目标标签不参与训练。

| 协议 | 无标签目标池 | 评价对象 |
|---|---|---|
| Protocol 1 / Setting1 | B2–B6 合并 | 分别评价 B2–B6 |
| Protocol 2 / Setting2 | 当前 Bk | Bk，k=2,…,6 |
| Protocol 3 / Setting3 | B2,…,Bk | Bk，k=2,…,6 |

Protocol 1 每个种子训练一个模型；Protocols 2、3 每个评价任务重新初始化。Protocol 3 累积数据，不继承前一任务的模型。

| 参数 | 默认值 |
|---|---|
| 输入／类别／特征隐藏维度 | 54／3／200 |
| 特征提取器 | Linear → BatchNorm → ReLU；源、目标联合 BN |
| 域判别器隐藏层／谱范数上限 | 32 → 32／0.75 |
| 每域 minibatch／更新次数 | 20／300（60 × 5） |
| 优化器／学习率／weight decay | Adam／0.005（固定）／0.0001 |
| 域损失／LC 权重 | 0.9／1.2；DANN、CDAN 的 LC 权重为 0 |
| GRL | gamma=3，前 150 步递增，后 150 步固定 |
| 源域与目标域采样 | 随机加载器，每记录区间重启，丢弃不足批次 |

**当前 ST 默认随机采样，没有固定 4/8/8 类别配额；与此前分层源域采样的论文实验不同。** `epoch` 是固定 5 次更新的记录区间，不表示完整遍历数据，优化器状态在 300 步内保持连续。

预处理为 asinh 后按源域统计量做 z-score。初始化在网络构建后按种子重置。DANN 使用特征作域输入，CDAN 使用特征与 detached 类别概率的外积。LC 对齐源类中心和目标软概率加权中心，只计入源小批次出现的类别。域判别器在初始化及每次更新后进行谱范数投影。

## 输出与统计

默认输出至 `dann_cdan_lc/results/ST/`，按种子、任务、模型保存 JSON 和最终 `.pt` 权重；相同路径重跑会覆盖同名文件。评估使用最后一步模型。`test.py` 汇总已有结果，不重新训练。

每个种子先对五个目标批次准确率等权平均，再计算种子间均值和样本标准差（ddof=1）。JSON 保存逐种子均值、均值／SD 和配对 LC 增益；CSV 保存均值和平均增益。只运行一个种子时，SD 为 `null`；仅当同系列两个模型都被选中时计算 LC 增益。

目标标签不用于反向传播或选择 checkpoint；参数与预算曾参考目标评估结果选择，因此比较属于探索性实验。默认结果、权重与缓存已被 `.gitignore` 排除，自定义输出建议放在仓库外。

## 代码结构

| 文件 | 职责 |
|---|---|
| main.py | 实验入口和任务循环 |
| loaddataset.py | 读取数据、预处理和协议定义 |
| params.py | 参数与模型选择 |
| model.py | 网络、初始化、GRL 和谱范数投影 |
| train.py | 损失计算与优化 |
| test.py | 准确率评估和统计汇总 |
| utils.py | 采样与结果保存 |
