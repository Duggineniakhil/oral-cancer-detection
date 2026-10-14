"""
Summarize multi-seed runs as mean ± SD per setting.

Single training runs of the same setting differed by 3-5 points, so
results are reported over several seeds. Run folders are expected to
be named <setting>_seed<N> (as created by the multi-seed notebook),
e.g. runs/convnext_tiny_3class_224px_seed1/metrics.json.

Usage:
    python -m scripts.classification.summarize_seeds

Outputs (OUTPUT_ROOT/runs/):
    multiseed_per_run.csv      one row per run
    multiseed_summary.csv      mean and SD per setting
    multiseed_summary.md       the same as a Markdown table (for the README)
"""

import json
import re

import numpy as np
import pandas as pd

from oral_cancer.config import OUTPUT_ROOT


# ============================================================
# CONFIGURATION
# ============================================================

RUNS_DIR = OUTPUT_ROOT / "runs"
RUN_PATTERN = re.compile(r"^(?P<setting>.+)_seed(?P<seed>\d+)$")

SETTING_LABELS = {
    "convnext_tiny_3class_224px": "3-class, NDB-UFES (primary)",
    "convnext_tiny_4class_224px": "4-class",
    "convnext_tiny_4class_gray_224px": "4-class, greyscale",
}

CLASSES = [
    "OSCC",
    "Leukoplakia_with_dysplasia",
    "Leukoplakia_without_dysplasia",
    "Normal",
]

# Columns shown in the Markdown table, in order
REPORT_COLUMNS = [
    ("balanced_accuracy", "Balanced acc."),
    ("macro_f1", "Macro-F1"),
    ("accuracy", "Accuracy"),
    ("macro_auc_ovr", "Macro AUC"),
    ("f1_Normal", "Normal F1"),
    ("ndb_balanced_accuracy", "NDB-UFES-only bal. acc."),
]


# ============================================================
# COLLECT
# ============================================================

def collect():
    rows = []

    for metrics_file in sorted(RUNS_DIR.glob("*/metrics.json")):
        match = RUN_PATTERN.match(metrics_file.parent.name)
        if not match:
            continue

        m = json.load(open(metrics_file))
        t = m["test"]

        row = {
            "setting": match["setting"],
            "seed": int(match["seed"]),
            "best_epoch": m["best_epoch"],
            "accuracy": t["accuracy"],
            "balanced_accuracy": t["balanced_accuracy"],
            "macro_f1": t["macro_f1"],
            "macro_auc_ovr": t["macro_auc_ovr"],
        }
        for c in CLASSES:
            if c in t["per_class"]:
                row[f"f1_{c}"] = t["per_class"][c]["f1-score"]
        if "ndb_ufes_subset" in t:
            row["ndb_balanced_accuracy"] = t["ndb_ufes_subset"]["balanced_accuracy"]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    per_run = collect()

    if per_run.empty:
        print(f"No runs named <setting>_seed<N> found in {RUNS_DIR}")
        return

    per_run = per_run.sort_values(["setting", "seed"])
    per_run.to_csv(RUNS_DIR / "multiseed_per_run.csv", index=False)

    metrics = [c for c in per_run.columns if c not in ("setting", "seed", "best_epoch")]
    grouped = per_run.groupby("setting")[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    summary.insert(0, "seeds", per_run.groupby("setting")["seed"].count())
    summary.to_csv(RUNS_DIR / "multiseed_summary.csv")

    # --------------------------------------------------------
    # Markdown table: mean ± SD in percent (AUC as 0-1)
    # --------------------------------------------------------
    order = [s for s in SETTING_LABELS if s in summary.index] + \
            [s for s in summary.index if s not in SETTING_LABELS]

    header = "| Setting | Seeds | " + " | ".join(label for _, label in REPORT_COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(REPORT_COLUMNS) + 2)]

    for setting in order:
        cells = []
        for key, _ in REPORT_COLUMNS:
            mean = summary.loc[setting].get(f"{key}_mean", np.nan)
            sd = summary.loc[setting].get(f"{key}_sd", np.nan)
            if pd.isna(mean):
                cells.append("—")
            elif key == "macro_auc_ovr":
                cells.append(f"{mean:.3f} ± {sd:.3f}")
            else:
                cells.append(f"{mean * 100:.1f} ± {sd * 100:.1f}%")
        label = SETTING_LABELS.get(setting, setting)
        lines.append(f"| {label} | {int(summary.loc[setting, 'seeds'])} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    (RUNS_DIR / "multiseed_summary.md").write_text(table + "\n", encoding="utf-8")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("========================================")
    print("PER RUN (test set)")
    print("========================================")
    show = per_run[["setting", "seed", "best_epoch", "balanced_accuracy", "macro_f1", "accuracy"]].copy()
    for c in ("balanced_accuracy", "macro_f1", "accuracy"):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))

    print("\n========================================")
    print("MEAN ± SD OVER SEEDS")
    print("========================================")
    print(table)
    print(f"\nSaved to: {RUNS_DIR}")


if __name__ == "__main__":
    main()
