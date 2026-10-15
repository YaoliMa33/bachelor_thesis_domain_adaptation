# ST 自采数据

[项目首页](../README.md) · [训练说明](../st_training/README.md)

本数据集包含 303 条采集记录、3 类气体和 6 个批次。

| 文件 | 内容 |
|---|---|
| [st_54d/features_54d.csv](st_54d/features_54d.csv) | 已提取的 54 维特征，供训练直接读取 |
| [st_raw/st_raw_csv.zip](st_raw/st_raw_csv.zip) | 对应的 303 个原始时序 CSV |
| [st_raw/sample_manifest.csv](st_raw/sample_manifest.csv) | 样本、标签、批次与原始文件的对应关系 |

## 特征与标签

每路传感器提取 9 个描述量，6 路共 54 维；按文件顺序读取 `s1_*` 至 `s6_*`。描述量为 `exp_mean`、`exp_absmax`、`exp_span`、`exp_tail`、`exp_auc_abs`、`rec_mean`、`rec_tail`、`rec_slope`、`overall_std`。CSV 另有 9 个元数据列，共 63 列；标签和其他元数据不作为模型输入。

训练标签为 `air → 0`、`alcohol → 1`、`acetone → 2`。

| 批次 | CSV 中的 batch | 样本数 |
|---|---|---:|
| B1 | source | 30 |
| B2 | batch1 | 28 |
| B3 | batch2 | 84 |
| B4 | batch3 | 81 |
| B5 | batch4 | 46 |
| B6 | batch5 | 34 |

54 维特征已包含提取阶段的逐记录基线校正，尚未施加训练阶段的 asinh 或 z-score。训练加载器在内存中执行 `asinh → 源域拟合的 z-score`，不修改数据文件。

## 原始记录与许可

训练现有模型不需要解压原始 CSV。需要查看时序时，将 ZIP 解压至 `st_raw`，按清单的 `target_file` 定位；跨平台读取时转换路径分隔符。采集字段、原始标签与时间轴见 [原始数据说明](st_raw/README.md)。

本目录数据采用 [CC BY 4.0](LICENSE.md)，使用时注明来源、许可及修改情况。
