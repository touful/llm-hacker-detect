"""Render paper tables from frozen summaries; report exploratory client strata separately."""

import argparse
import json
from pathlib import Path
import numpy as np
from .ablation import VIEWS
from .study import load, scores
from .evaluate import _metrics


def render(root, cohort, pilot):
    def read(rel):
        return json.loads((root / rel).read_text(encoding="utf8"))

    z = load(cohort)
    dev = np.isin(z["files"], load(pilot)["files"])
    ix = np.flatnonzero(~dev)
    y = z["y"][ix]
    ds = read("development/summary.json")["models"]
    cs = read("confirmation/summary.json")["models"]
    lock = read("lock.json")
    comp = read("comparisons.json")
    null = read("permuted/summary.json")["models"]["dist-hgb"]["metrics"]
    lines = [
        "# 固定包预算下的来源归因：论文实验表",
        "",
        "协议见 [study-protocol.md](study-protocol.md)。每个 PCAP 最多前 2,048 条记录。开发 533 场，确认 4,795 场，沿用文件级原折。特征选择只在开发部分进行；确认部分重新训练三折、验证一折、测试一折，轮换五次，不是外部测试。以下 AP 为 average precision。",
        "",
        "## 表1　开发集特征块消融（同一个 HGB，每个输入重新拟合）",
        "",
        "|输入|维数|评分数|AP|相对完整 ΔAP|AUC|F1|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base = ds["dist-hgb"]["metrics"]["ap"]
    for name, r in ds.items():
        m = r["metrics"]
        lines.append(
            f"|{name}|{len(VIEWS[name]) if name in VIEWS else 2190}|{m['support']['scored']}|{m['ap']:.4f}|{m['ap'] - base:+.4f}|{m['auc']:.4f}|{m['f1']:.4f}|"
        )
    lines += [
        "",
        "raw：16 个原始箱；stats：跨箱最小、最大、加权均值及标准差；flags：90 个顺序出现标志；dev：各箱偏离；style：三段头风格。no-X 为仅删除该块；no-explicit-pairs 删除原始/统计中的显式顺序对及全部顺序标志，其他代理信息可能保留；field-only 只保留 11 个字段存在率的原始箱和四类统计。",
        "",
        f"预设候选中锁定 `{lock['choice']}`（{lock['dim']} 维）。候选规则是相对完整 AP 损失 ≤0.015 时选维数最小者，不是事后挑选最大的确认集分数。",
        "",
        "## 表2　确认集横向比较",
        "",
        "|模型|评分数|共同集 AP|自身 AP|自身 AUC|F1|MCC|Brier|实际 FPR/TPR（目标1%）|实际 FPR/TPR（目标5%）|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, r in cs.items():
        m = r["metrics"]
        ops = [
            f"{r['low_fpr'][k]['fpr']:.2%} / {r['low_fpr'][k]['tpr']:.2%}"
            for k in ("1", "5")
        ]
        lines.append(
            "|"
            + "|".join(
                [
                    name,
                    str(m["support"]["scored"]),
                    f"{r['common_support_metrics']['ap']:.4f}",
                    *[f"{m[k]:.4f}" for k in ("ap", "auc", "f1", "mcc", "brier")],
                    *ops,
                ]
            )
            + "|"
        )
    lines += [
        "",
        "## 表3　成对比较（仅共同可评分会话）",
        "",
        "|对比：前者减后者|场次|前者 AP|后者 AP|差值|条件 95% 区间|",
        "|---|---:|---:|---:|---:|---|",
    ]
    for k in ("compact_minus_full", "eof_minus_prefix"):
        r = comp[k]
        lines.append(
            f"|{k}|{r['n']}|{r['ap_first']:.4f}|{r['ap_second']:.4f}|{r['delta']:+.4f}|[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]|"
        )
    d = comp["duplication"]
    lines += [
        "",
        "第一行为确认集，第二行为开发集；分别训练两种预算的模型。按折和类别分层配对 bootstrap 1,000 次，固定已拟合模型，不涵盖重新训练或同题/同人聚类不确定性。",
        "",
        f"稳健性：确认集折内一次标签置乱 AP={null['ap']:.4f}，AUC={null['auc']:.4f}。确认集可评分的 {d['n_scored']} 行中，{d['cross_fold_rows']} 行的完整特征向量也见于另一确认折，{d['dev_overlap_rows']} 行见于开发集；排除二者并集 {d['excluded_union']} 行后，剩 {d['n_remaining']} 行，完整/精简 AP={d['ap_full_on_remaining']:.4f}/{d['ap_selected_on_remaining']:.4f}。此项只过滤评估行，不是去重重训；特征相同不代表 PCAP 字节相同。",
        "",
        "## 表4　按可见客户端构成分层（确认集，事后诊断）",
        "",
        "完整模型，不改阈值或重新训练。按请求中粗分类 browser / script 的比例分组；纯组要求该类占全部已解析请求。other 表示两个已知类别均未识别，mixed 表示非纯且至少一种已识别。分类基于 UA 粗分类，不能替代精确客户端或题目匹配。AP 随组内正类比例变化。",
        "",
        "|组|总人类/Agent|评分人类/Agent|评分集正类占比|AP|AUC|",
        "|---|---|---|---:|---:|---:|",
    ]
    p = scores(root / "confirmation/dist-hgb/oof.csv", z, ix)
    b, s = z["conv"][ix, 69:71].T
    groups = np.select(
        [~np.isfinite(b), b == 1, s == 1, (b == 0) & (s == 0)],
        ["unknown", "browser", "script", "other"],
        default="mixed",
    )
    for g in np.unique(groups):
        mask = groups == g
        m = _metrics(y[mask], p[mask])
        c = m["support"]["by_class"]
        a, h = c["1"], c["0"]

        def fmt(x):
            return "—" if x is None else f"{x:.3f}"

        lines.append(
            f"|{g}|{h['n']} / {a['n']}|{h['scored']} / {a['scored']}|{fmt(a['scored'] / m['support']['scored'] if m['support']['scored'] else None)}|{fmt(m['ap'])}|{fmt(m['auc'])}|"
        )
    lines += [
        "",
        "## 复跑",
        "",
        "在 attribution/ 目录，使用 Windows py312（sklearn "
        + lock["sklearn"]
        + "、NumPy "
        + lock["numpy"]
        + "）。已有项目内 533 场完整抓包缓存 runs/full-session-pilot.npz。使用新输出路径：",
        "",
        "```powershell",
        "$py='D:\\software\\program\\miniconda\\envs\\py312\\python.exe'",
        "& $py -m agenttrace extract --labels ../labels.csv --max-packets 2048 --out runs/new-all.npz",
        "& $py -m agenttrace.study runs/new-all.npz runs/full-session-pilot.npz --out runs/new-study",
        "& $py -m agenttrace.study_report runs/new-study runs/new-all.npz runs/full-session-pilot.npz --out runs/new-tables.md",
        "```",
        "",
        "各模型 summary.json、逐文件 oof.csv、含零起始列号与列名的 lock.json、comparisons.json 均位于 study 输出目录。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for k in ("root", "cohort", "pilot"):
        p.add_argument(k, type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with a.out.open("x", encoding="utf8", newline="\n") as f:
        f.write(render(a.root, a.cohort, a.pilot))
