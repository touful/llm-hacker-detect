# 模型注册表与实验角色

- **正文完整模型**：Dist24-HGB，即 `dist-hgb`；**精简模型**是同一个 HGB 的 `flags` 输入投影，只保留请求头的 90 个有向先后标志。每个标志记录两个字段在同一请求内的相对顺序是否出现过。
- **研究专用名称**（如 `flags`、`no-explicit-pairs`）：由 [ablation.py](../agenttrace/ablation.py) 定义列集合，经 [study.py](../agenttrace/study.py) 的工厂交给评估器，入口为 `agenttrace.study`。确认集实际运行的是 flags、dist-hgb、dist-et、dist-gbdt、dist-linear、behavior-hgb。
- **传统模型矩阵**：`evaluate --classical`；十种分类器各有 conv/dist 两种输入。具体标记和参数见[传统模型附表](classical.md)。
- **默认模型**：下表列出 `evaluate` 默认运行的 11 个模型。通用入口 `create(name, seed, epochs, batch_size)`；fit 只看训练折，predict 返回正类分数，拒评为 NaN。


|名称|实验变量|
|---|---|
|conv-hgb|145 维直接拼接。HGB 80 轮，学习率 .05，最多 15 叶，最小叶 20，L2=1，关闭早停。|
|dist-hgb|2,190 维；与 conv-hgb 同一个分类器，隔离表示差异。|
|dist-et|同一个 Dist24 输入，96 棵极端随机树，max_features=.8，最小叶 5，无深度上限。|
|header-mil|每箱 619 → 32；共享残差 MLP；带位置的注意力、最大与均值三池化；446 → 32 → 残差 → 1；38,434 个可训练参数。|
|header-tcn|同样的 619 → 32；三层残差卷积，核 3、膨胀 1/2/4、对称填充；每层重新置空箱为零；最大+均值池化；414 → 32 → 残差 → 1；44,577 个参数。对称卷积同时使用当前箱两侧的上下文。|
|header-gru|共享同样的比例编码、位置及辅助分支，GRUCell 只在有效箱更新；取最后状态，40,577 个参数。检验递归编码是否有益。|
|dist-linear|相同 Dist24 输入、中位数填充、标准化、L2 logistic (C=1)。作为线性可分性对照。|
|bag-hgb|502 维，即 4×103 跨箱统计和 90 个出现标志。汇总后保留箱内结构切换信息，移除箱位置和时间风格。|
|behavior-hgb|56 维，不读 HTTP 特征：活动 24 维、前缀方向/长度 6 维、流级除客户端两项外的 26 维。|
|joint-hgb|早期融合，即 conv 和 dist 拼接后训练一棵提升树集成。|
|view-fusion|后期融合，即 behavior-hgb 与 dist-hgb 的分数等权平均；单一分支拒评时只用可用分支，两者都拒评才拒评；不学习测试权重。|

对表格模型先删除训练折全空列。其余缺失用 HGB 原生路由，ET/线性模型用训练列中位数。测试样本在所有受支持列上都无观测则拒评。神经网络空箱按 NaN 判断，不按 0 判断；若全场无有效箱则拒评。六通道依次为值/有效/零/一/截断负对数/对补数的截断负对数：截断下界 1e-6，用 log(1e6) 缩放。未知项六通道全零；最后 206 维出现标志和 144 维风格编码不因空箱生成伪观测。网络输出 logits，训练用 BCEWithLogitsLoss，推断用 sigmoid。AdamW 学习率 .001、默认权重衰减 .01、12 个固定 epoch、batch 32，不在测试集选 epoch；设备 cuda:0。这些参数用于开发集神经网络对照。
