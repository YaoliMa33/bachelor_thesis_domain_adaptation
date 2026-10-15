# UCI 128 维特征数据

[项目首页](../README.md) · [训练说明](../uci_training/README.md)

[Gas Sensor Array Drift at Different Concentrations（UCI 270）](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) 包含 13,910 条记录、6 类气体和 10 个批次；每条记录为 16 个传感器各 8 个描述量，共 128 维。

## 文件格式与标签

`batch1.dat` 至 `batch10.dat` 保留官方文件内容，格式为：

```text
label; concentration 1:value ... 128:value
```

标签和浓度不作为输入特征。训练时将数字标签减 1，并执行 asinh 与源域拟合的 z-score；不修改磁盘文件。

| 文件标签 | 训练索引 | 气体 |
|---:|---:|---|
| 1 | 0 | Ethanol |
| 2 | 1 | Ethylene |
| 3 | 2 | Ammonia |
| 4 | 3 | Acetaldehyde |
| 5 | 4 | Acetone |
| 6 | 5 | Toluene |

[gas_labels.json](gas_labels.json) 保存上述对应关系，依据官网明确的数字标签说明。官网历史样本数量表的排列存在冲突，本项目未据此重标原始文件。

## 来源与许可

数据引用：Vergara, A. (2012). *Gas Sensor Array Drift at Different Concentrations*. UCI Machine Learning Repository. [DOI: 10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M)。官网同时要求引用：

- Vergara et al. (2012). *Chemical gas sensor drift compensation using classifier ensembles*. Sensors and Actuators B: Chemical.
- Rodriguez-Lujan et al. (2014). *On the calibration of sensor arrays for pattern recognition using the minimal number of experiments*. Chemometrics and Intelligent Laboratory Systems, 130.

截至 2026-10-16，官网 License 栏列为 CC BY 4.0，旧版 Additional Information 仍写有“仅限研究、排除商用”。本项目保留两种说明，不自行解决其冲突或授予额外权利；将数据视为可无限制再分发或商用前，应向提供方确认。本项目代码的 MIT 和 ST 数据的 CC BY 4.0 不替代 UCI 数据授权。
