# AgentTrace：Web CTF 解题会话的网络来源归因

给定一份 Web CTF 抓包，判断该场解题由人类（0）还是 AI Agent（1）主导。样本单位是 **PCAP 文件**，不是请求、题目或操作者。输出用于筛选值得复核的会话，不直接认定个人行为。

**写论文从[写作交接](docs/writing-guide.md)开始，取数用[主实验表](docs/study.md)。** 方法定义、历史附表和复跑入口如下。只阅读本仓库即可理解正文结果；重算指标或训练还需本地运行产物或原始数据。

|要回答的问题|文档|对应实现|
|---|---|---|
|样本如何划分？特征何时选定？|[运行前固定的协议](docs/study-protocol.md)|[study.py](agenttrace/study.py)|
|抓包能看见什么？2,190 列代表什么？|[字段、分箱与可见性](docs/features.md)|[capture.py](agenttrace/capture.py) → [features.py](agenttrace/features.py) / [views.py](agenttrace/views.py)|
|主模型、精简模型与基线是什么关系？|[模型与入口](docs/models.md)|[models.py](agenttrace/models.py)、[ablation.py](agenttrace/ablation.py)、[classical.py](agenttrace/classical.py)、[networks.py](agenttrace/networks.py)|
|低误报阈值、共同支持集和拒评怎么算？|[指标定义](docs/metrics.md)|[evaluate.py](agenttrace/evaluate.py)|
|正文中的表格来自哪里？|[主实验表](docs/study.md)：开发消融、确认、成对比较、客户端分层|[study_report.py](agenttrace/study_report.py) 从现有结果生成，不训练模型|
|早期还比较过什么？|[开发集传统模型矩阵](docs/classical.md)、[开发集树与神经网络](docs/pilot.md)|[命令行](agenttrace/__main__.py) 的 `evaluate --classical`、`evaluate --models`|
|哪些文献支持任务或方法设计？|[文献笔记](docs/literature.md)|定向调研，不是本项目的测量结果|

## 两种观察预算，不是两份独立数据集

`../labels.csv` 和 `../pcap/` 是数据，不在代码仓库中。全库 **5,328 场的前缀**（每场最多 2,048 条原始记录）已提取。533 场开发集用于选特征；其余 4,795 场在原五折内重新训练与测试。**整场读至 EOF** 的实验只覆盖这 533 场；没有全库 EOF、跨题目或跨框架验证。

## 复核与复跑（本机 PowerShell，在 attribution/ 中）

- **只复核数字**：本地 `runs/paper-study/` 存主研究的 `summary.json`、逐模型 `oof.csv`、选择文件 `lock.json`、成对比较 `comparisons.json`。这些不随 Git 分发；已跟踪的 `docs/study.md` 是报告快照。重生报告不需要重扫 PCAP：

```powershell
$paper='D:\software\program\miniconda\envs\py312\python.exe'
& $paper -m agenttrace.study_report runs/paper-study runs/cohort2048.npz runs/full-session-pilot.npz --out runs/check-tables.md
```

- **重做主研究**：[主实验表末尾](docs/study.md)给出提取、研究和生成报告的命令。主实验用 Python 3.12、NumPy 2.4.3、scikit-learn 1.9.0，仅 CPU；若没有 533 场 EOF 缓存，先按下方历史环境的 `extract` 命令生成，再将该路径传给 `study` 和 `study_report`。
- **重做开发集附表**：[requirements.txt](requirements.txt) 记录的是 Python 3.11 历史参考环境（NumPy 2.0.1、scikit-learn 1.8.0；神经网络用 CUDA），不是主研究的环境锁。`extract`、`evaluate`、`study`、`study_report` 均使用新输出路径，不覆盖现有结果。

```powershell
$py='D:\software\program\miniconda\envs\torch_311\python.exe'
& $py -m agenttrace extract --labels ../labels.csv --fraction 0.10 --seed 42 --out runs/repro-eof.npz
& $py -m agenttrace evaluate runs/repro-eof.npz --classical --out runs/repro-classical
& $py -m unittest discover -s tests -v
```

每个报告使用的缓存、环境和可引用范围见[写作交接](docs/writing-guide.md)。不要把不同运行目录下的最好成绩拼成一张排行榜。
