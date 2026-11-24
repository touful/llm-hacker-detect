"""从 evaluate 的全精度结果输出传统模型附表，或核对已发布的 Markdown。"""

import argparse
import json
from pathlib import Path

HEADER = "|配置|AP|ROC-AUC|F1|MCC|Brier|精确率|召回率|目标1%：FPR/TPR|目标5%：FPR/TPR|"
METRICS = ("ap", "auc", "f1", "mcc", "brier", "precision", "recall")


def table(models):
    rows = [HEADER, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in models.items():
        common = result["common_support_metrics"]
        cells = [name, *(f"{common[k]:.3f}" for k in METRICS)]
        for target in ("1", "5"):
            op = result["low_fpr"][target]
            cells.append(" / ".join(f"{op[k]:.1%}" for k in ("fpr", "tpr")))
        rows.append("|" + "|".join(cells) + "|")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--check", type=Path, help="核对现有附表，不写文件")
    args = parser.parse_args()
    expected = table(json.loads(args.summary.read_text(encoding="utf-8"))["models"])
    if args.check is None:
        print(expected)
        return
    doc = args.check.read_text(encoding="utf-8")
    matches = [block for block in doc.split("\n\n") if block.startswith(HEADER)]
    if matches != [expected]:
        parser.exit(1, "Markdown table differs from frozen summary\n")
    print("OK: table matches frozen summary")


if __name__ == "__main__":
    main()
