# 文献与可检验的设计选择

2026-11-23，使用公开网页检索（Tavily），复核下列 arXiv 摘要、PMLR 论文页与作者页。关键词覆盖 human vs LLM agent CTF traffic attribution、HTTP header order web bot detection、fingerprinting AI browsing agents、attention multiple instance learning、temporal convolutional recurrent networks。定向调研，非系统综述或新颖性证明。没有向外部服务传送抓包。

|来源|任务及其对本项目的意义|
|---|---|
|[NYU CTF Bench](https://arxiv.org/abs/2406.05590) (2024，v3 2025)|对 LLM 解 CTF 的能力与工具交互做评测。它支持把一次解题作为研究单位；解题成功率并不是来源分类准确率。|
|[FP-Agent: Fingerprinting AI Browsing Agents](https://arxiv.org/abs/2605.01247) (2026 预印本)|受控网站上比较七种浏览 Agent 与人类，区分浏览器指纹与行为指纹；鼠标、键盘和滚动需要网站插桩，本项目不能从 PCAP 声称观测到这些信号。启发：同客户端对照，比单看 OOF AUC 更能检验归因解释。|
|[When Handshakes Tell the Truth](https://arxiv.org/abs/2602.09606) (2026 预印本)|用 TLS 指纹和提升树检测 bad bots，并非 HTTP 分箱或 LLM 归因实验。提示：强客户端实现特征能让树模型表现很好，也构成重要混杂。|
|Ilse, Tomczak & Welling, [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html) (ICML 2018)|只有袋标签，没有实例标签，与 PCAP 有标签而请求箱无标签相符。此处加入箱位置和辅助分支，不是原论文模型的逐层复刻；注意力也不是因果证据。|
|Bai, Kolter & Koltun, [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://vladlen.info/publications/tcn/) (2018)|为同输入下比较卷积和递归提供依据。这里 TCN 与 GRU 共享编码和掩码；因本任务是离线取证，TCN 使用对称卷积。通用序列任务的名次不能外推到 CTF。|

实验解释顺序：先比较同输入的线性、HGB 和 ExtraTrees，再比较共享输入编码的 MIL、TCN、GRU；用行为、去箱位置、早融合与晚融合检查信号和覆盖率差异。尚待验证：同题同客户端、留一题目、留一 Agent 框架、独立采集批次。未有分组信息时不声称跨域有效。
