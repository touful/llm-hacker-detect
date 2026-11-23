# AgentTrace：Web CTF 会话来源分类

二分类：0 人类主导，1 AI Agent 主导，每个 PCAP 是一个样本。项目只离线读本地 PCAP，输出聚合特征、概率、指标。不要用分数单独认定个人行为。

## 已有结果

已从 5,328 场按 label × fold 以 seed 42 抽 10%（533 场，362/171）。相同 533 场读到 EOF、测试 11 个模型；另有前 2,048 条原始记录的 9 模型预算对照。见 [指标表](docs/pilot.md)、[文献](docs/literature.md)、[列口径及解析边界](docs/features.md)、[模型](docs/models.md)、[指标及划分](docs/metrics.md)。不把原 fold 当作已知题目分组。实现遵循提供的尺寸、结构及树参数；未提供的列定义与训练设置在文档中显式固定，不等价于前期成绩复现。

## 本机复跑（PowerShell，先进入 attribution/）

```powershell
$py='D:\software\program\miniconda\envs\torch_311\python.exe'
& $py -m unittest discover -s tests -v
& $py -m agenttrace extract --labels ../labels.csv --fraction 0.10 --out runs/repro.npz
& $py -m agenttrace evaluate runs/repro.npz --out runs/repro-report
```

输出必须使用新路径。全库版去掉 --fraction 0.10；若采用包预算，向 extract 传 --max-packets 2048 并独立命名缓存。树模型可仅传 --models dist-hgb dist-et；神经网络用 CUDA。Python 3.11，见 requirements.txt。缓存只存特征/标签/折号/文件标识/各类事件数量及提取参数，可多次更换模型而不必再读抓包。正式全库尚未执行；先检查最大文件内存。每场计数及数据集索引不输入分类器，包数等显式行为特征除外。

已有本地运行明细：runs/full-session-report/（九个完整抓包模型）、runs/full-extra/（两个补充对照）、runs/pilot-report/（九个包预算模型）。summary.json 含各折及汇总结果；每模型 oof.csv 保留未评分行。代码与摘要纳入本地版本控制，特征缓存、逐文件输出及原始数据不纳入。
