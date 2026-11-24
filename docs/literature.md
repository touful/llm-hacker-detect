# 相关工作：解题能力、客户端指纹与会话归因

[NYU CTF Bench](https://arxiv.org/abs/2406.05590) 以解题结果评估 LLM 使用工具完成 CTF 挑战的能力。网络取证面对的是另一个问题：给定一次解题留下的抓包，哪些交互痕迹有助于识别会话来源？这里的监督标签属于整份抓包。

## 观察位置决定能提取的信号

[FP-Agent](https://arxiv.org/abs/2605.01247) 在受控网站上比较七种浏览 Agent 与人类，通过网站插桩记录键盘、鼠标和滚动行为。[When Handshakes Tell the Truth](https://arxiv.org/abs/2602.09606) 从网络协议取证，用 TLS 指纹和提升树区分机器人流量。两项工作的观察位置不同，也提示了两种可能的区分依据：操作者的交互方式，以及客户端实现留下的协议特征。Web CTF 抓包能记录明文 HTTP 请求头的结构，因此[主实验表 4](study.md)按粗客户端类别分层报告归因结果。

## 会话标签与请求箱的聚合

[Ilse、Tomczak 与 Welling 的注意力多实例学习方法](https://proceedings.mlr.press/v80/ilse18a.html)从袋级标签学习实例的聚合权重。在会话归因中，一份抓包对应一个袋，请求箱是实例，HeaderDist-MIL 汇总各箱表示后输出会话分数。[Bai、Kolter 与 Koltun 对卷积与递归模型的比较](https://vladlen.info/publications/tcn/)提供了另一种检验思路：固定输入，改变序列编码器。[开发集附表](pilot.md)据此在相同的分箱、位置和掩码上比较 TCN 与 GRU；TCN 使用对称卷积，以适配离线取证。
