"""Render experimental tables from round-2 JSON files, without refitting."""

import json
from pathlib import Path


def render(root):
    def read(folder):
        return json.loads((root / folder / "summary.json").read_text("utf8"))

    f = read("fusion-fallback")["models"]
    d = read("dedup-retrain")
    s = read("slr-exact")
    rows = [
        "# 确认集取证补充实验：多视图、去重重训与分数校准",
        "",
        "在 1,536 场 Agent 会话中，验证目标 5% FPR 下，联合模型检出 934 场，Dist24 检出 860 场。以下为同一确认库上的事后比较。开发集 533 场，确认集 4,795 场；每场最多 2,048 条原始记录。原折号轮换三折训练、一折验证、一折测试。",
        "",
        "## 多视图：[summary.json](fusion-fallback/summary.json)",
        "",
        "|模型|HTTP 共同集 AP / AUC|自身评分数|自身集 AP / AUC|1% 目标：实测 FPR / TPR|5% 目标：实测 FPR / TPR|全部 Agent 检出数（1% / 5%）|",
        "|---|---|---:|---|---|---|---|",
    ]
    for n, x in f.items():
        h = x["cohorts"]["http"]["metrics"]
        a = x["cohorts"]["all"]
        m = a["metrics"]
        p = a["low_fpr"]
        rows.append(
            f"|{n}|{h['ap']:.4f} / {h['auc']:.4f}|{m['support']['scored']}|{m['ap']:.4f} / {m['auc']:.4f}|"
            + "|".join(f"{p[t]['fpr']:.2%} / {p[t]['tpr']:.2%}" for t in ("1", "5"))
            + f"|{p['1']['confusion']['tp']} / {p['5']['confusion']['tp']}|"
        )
    rows += [
        "",
        "低误报率在各自有分数的测试会话上计算，阈值由验证折决定。全体 Agent 检出率的分母始终为 1,536。级联优先用验证折校准的 Flags 分数；188 场回退至传输分支，58 场拒评。在 5% 目标下，HTTP 分支检出 846 场、误报 143 场；回退分支检出 55 场、误报 20 场。各模型逐行记录在 `fusion-fallback/<模型>/oof.csv`。JSON 含按折和标签配对自助抽样 1,000 次的共同集 AP 差值区间（种子 2026，固定折外预测）。",
        "",
        "## 先去重、再重训：[summary.json](dedup-retrain/summary.json)",
        "",
        "|去重规则|模型|评分数|AP|AUC|Brier|1% 目标：实测 FPR / TPR|5% 目标：实测 FPR / TPR|",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for rule in ("strict", "representative"):
        for n, x in d[rule]["models"].items():
            m = x["metrics"]
            p = x["low_fpr"]
            rows.append(
                f"|{rule}|{n}|{m['support']['scored']}|{m['ap']:.4f}|{m['auc']:.4f}|{m['brier']:.4f}|"
                + "|".join(f"{p[t]['fpr']:.2%} / {p[t]['tpr']:.2%}" for t in ("1", "5"))
                + "|"
            )
    rows += [
        "",
        "严格集删除与开发集或其他确认折向量相同的全部 473 行；代表集剔除与开发集相同的向量，其余非空哈希簇各留文件编号最小的一行。两组均保留 246 场空向量。保留标记见 `dedup-retrain/audit.csv`。",
        "",
        "## 同源验证分数校准：[summary.json](slr-exact/summary.json)",
        "",
        "|模型|校准|Cllr|Cllr min|Cllr cal|ECE15|MCE15|Agent: LR ≥10 / ≥100|人类: LR <1 / ≤0.1|",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for n in ("dist", "flags"):
        for method in ("platt", "laplace_pav"):
            m = s[n][method]
            t = m["tippett"]
            rows.append(
                f"|{n}|{method}|"
                + "|".join(
                    f"{m[key]:.4f}"
                    for key in (
                        "cllr_bits",
                        "cllr_min_bits",
                        "cllr_cal_bits",
                        "ece15",
                        "mce15",
                    )
                )
                + f"|{t['agent_ge10']:.2%} / {t['agent_ge100']:.2%}|{t['human_lt1']:.2%} / {t['human_le01']:.2%}|"
            )
    rows += [
        "",
        "校准器只用同一模型在下一验证折的预测；用该分支验证折可评分样本的先验赔率将后验换为 LR。Cllr min 取各校准方法输出的合并测试 log LR 排序上的最优单调代价；它用于描述分解，不用于预测。ECE/MCE 对后验使用 15 个等宽箱。CSV 的 `elub_pav` 列实际采用 Laplace 端点界限，不是经验 ELUB。",
        "",
        "[逐会话后验、原始分数和 log10 LR](slr-exact/oof_slr.csv)；[Tippett 曲线](slr-exact/tippett.csv)；[同源验证分数](validation_scores.csv)；[校准参数](calibrator_parameters.json)；[环境](environment.json)。新分解、后验可靠性与曲线应成组引用，不与旧补充报告的分解项拼接。",
        "",
        "实验 1（针对 Agent 的请求级伪装）未运行，无代理条件下的 AP 或检出率估计。",
        "",
        "复跑：在 attribution/ 下用 Windows PowerShell 运行 `& D:/software/program/miniconda/envs/py312/python.exe -B -m scripts.ifoss_round2 --out runs/repro-round2`，输出目录必须不存在。",
    ]
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1])
    text = render(root)
    with (root / "README_RESULTS.md").open("x", encoding="utf8") as f:
        f.write(text)
