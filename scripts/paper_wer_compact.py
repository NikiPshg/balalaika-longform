#!/usr/bin/env python3
"""Lay out the completed WER curves in one row, without recomputing metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(report, output):
    marker = json.loads((report / "REPORT_COMPLETE.json").read_text())
    source = report / "statistics/statistics.json"
    if (marker["state"] != "complete_fixed_subset" or marker["arm_group"] != "main13"
            or digest(source) != marker["output_sha256"]["statistics/statistics.json"]):
        raise ValueError("A complete, unchanged main13 report is required")
    payload = json.loads(source.read_text())
    backbones = {"cosy": "CosyVoice3", "qwen": "Qwen3-TTS", "vox": "VoxCPM2", "f5": "F5-TTS"}
    styles = {"base": ("Base", "#4C78A8", "o"),
              "short": ("Short-SFT", "#E28B31", "s"),
              "long": ("Long-SFT", "#218C74", "^"),
              "long_punct": ("Long-SFT-Punct", "#9467BD", "v")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.85), sharey=True)
    points, handles, upper = [], {}, []
    for axis, (backbone, title) in zip(axes, backbones.items()):
        for variant in styles:
            arm = backbone + "_" + variant
            if arm not in payload["cells"]["B4"]["arms"]:
                continue
            if arm == "cosy_long":
                continue  # Keep this separate ablation in the result table.
            display_variant = "long_punct" if backbone != "cosy" and variant == "long" else variant
            label, color, symbol = styles[display_variant]
            metrics = [payload["cells"][b]["arms"][arm]["metrics"]["wer_all_pct"]
                       for b in ("B0", "B2", "B4")]
            means = [m["estimate"] for m in metrics]
            bounds = [m["intervals"]["source_conditional"] for m in metrics]
            handles[label], = axis.plot(range(3), means, color=color, marker=symbol,
                                       linewidth=1.2, markersize=3.4)
            axis.vlines(range(3), [b["lo"] for b in bounds], [b["hi"] for b in bounds],
                        color=color, linewidth=.9, alpha=.7)
            upper.extend(b["hi"] for b in bounds)
            points.extend({"arm": arm, "bucket": b, "wer": m["estimate"],
                           "interval": m["intervals"]["source_conditional"]}
                          for b, m in zip(("B0", "B2", "B4"), metrics))
        axis.set_title(title, fontsize=9)
        axis.set_xticks(range(3), ["75", "300", "1200"])
        axis.grid(axis="y", alpha=.2)
        axis.tick_params(axis="both", labelsize=8)
    if len(points) != 36 or len({point["arm"] for point in points}) != 12:
        raise ValueError("The 12 displayed arms must retain all three input lengths")
    axes[0].set_ylabel("WER (%)")
    axes[0].set_ylim(0, max(upper) * 1.08)
    fig.supxlabel("Input length (words)", y=.20, fontsize=8)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=3,
               frameon=False, fontsize=8, handlelength=1.7, columnspacing=1.5)
    fig.subplots_adjust(left=.075, right=.99, top=.84, bottom=.37, wspace=.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=240, bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "source": str(source), "source_sha256": digest(source),
        "script_sha256": digest(Path(__file__)),
        "change": "presentation only; punctuated long arms share the Long-SFT-Punct label and style; all means and intervals unchanged",
        "omitted_from_figure": ["cosy_long"],
        "omitted_arm_results_retained_in": "Table 3 and the full source report",
        "points": points}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    render(args.report.resolve(), args.output.resolve())
