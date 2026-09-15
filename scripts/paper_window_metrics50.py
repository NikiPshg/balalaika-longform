#!/usr/bin/env python3
"""Plot the newly scored fixed50 window diagnostics without mixing older sets."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKBONES = {"cosy": "CosyVoice3", "qwen": "Qwen3-TTS", "vox": "VoxCPM2", "f5": "F5-TTS"}
VARIANTS = {"base": "Base", "short": "Short-SFT", "long": "Long-SFT", "long_punct": "Long-SFT-Punct"}
MAIN = ["cosy_base", "cosy_long_punct", "qwen_base", "qwen_long",
        "vox_base", "vox_long", "f5_base", "f5_long"]
COLORS = {"cosy_base": "#5a9ade", "cosy_short": "#9bc2ec", "cosy_long": "#205da7",
          "cosy_long_punct": "#15996f", "qwen_base": "#e5a23b", "qwen_short": "#eec878",
          "qwen_long": "#af640d", "vox_base": "#b187d5", "vox_short": "#d0b1ea",
          "vox_long": "#623889", "f5_base": "#e084aa", "f5_short": "#eda9c5",
          "f5_long": "#9a123e"}
KEY = ["set", "system", "text_id", "voice_id"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label(system):
    backbone, variant = system.split("_", 1)
    if backbone != "cosy" and variant == "long":
        variant = "long_punct"
    return BACKBONES[backbone] + " " + VARIANTS[variant]


def curves(windows, metric):
    data = windows.copy()
    data["time_sec"] = np.floor((data.t_center - 1.25 + 1e-9) / 2.5) * 2.5 + 2.5
    per_item = data.groupby(KEY + ["time_sec"], as_index=False)[metric].mean()
    return (per_item.groupby(["system", "time_sec"])[metric].agg(["mean", "count"])
            .reset_index().rename(columns={"mean": "raw", "count": "n"}))


def smooth(values):
    out = np.empty(len(values))
    for i, value in enumerate(values):
        out[i] = value if i == 0 else .3 * value + .7 * out[i-1]
    return out


def draw(mos, sim, systems, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(9.4, 3.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.6, .85], hspace=.07, wspace=.21)
    axes = [(fig.add_subplot(grid[0, i]), fig.add_subplot(grid[1, i])) for i in range(2)]
    handles = {}
    max_time = max(float(mos.time_sec.max()), float(sim.time_sec.max()))
    for data, (axis, support), ylabel, title, ylim in zip(
            (mos, sim), axes, ("DistillMOS proxy", "WeSpeaker cosine"),
            ("(a) Acoustic-quality proxy", "(b) Speaker similarity"), ((1, 5), (0, 1))):
        for system in systems:
            row = data[data.system == system].sort_values("time_sec")
            visible = row[row.n >= 2]
            if visible.empty:
                continue
            style = "--" if system.endswith("_base") else ":" if system.endswith("_short") else "-"
            axis.plot(visible.time_sec, visible.raw, color=COLORS[system], linestyle=style,
                      linewidth=.6, alpha=.22)
            handles[system], = axis.plot(visible.time_sec, smooth(visible.raw.to_numpy()),
                    color=COLORS[system], linestyle=style, linewidth=1.5)
            support.plot(row.time_sec, row.n, color=COLORS[system], linestyle=style,
                         linewidth=.9, drawstyle="steps-post")
        axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=10)
        axis.tick_params(labelbottom=False)
        support.set_ylim(0, 23)
        support.set_yticks([0, 11, 22])
        support.set_ylabel("Items")
        support.set_xlabel("Time (s)")
        for ax in (axis, support):
            ax.set_xlim(0, np.ceil(max_time/100)*100)
            ax.grid(alpha=.2, linewidth=.4)
    order = [s for s in systems if s in handles]
    fig.legend([handles[s] for s in order], [label(s) for s in order],
               loc="upper center", bbox_to_anchor=(.52, 1.12), ncol=4,
               frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.81)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix("." + extension), dpi=240, bbox_inches="tight")
    plt.close(fig)


def render(run, output):
    tasks = pd.DataFrame(read_rows(run / "tasks.jsonl"))
    expected = set(map(tuple, tasks[KEY].to_numpy()))
    if len(tasks) != 650 or len(expected) != 650 or set(tasks["set"]) != {"published50"}:
        raise ValueError("Require precisely the new 650-attempt task grid")
    if set(tasks.system) != set(COLORS) or set(tasks.groupby("system").size()) != {50}:
        raise ValueError("Require all 13 conditions with 50 attempts each")
    b4 = tasks[tasks.bucket == "B4"]
    if set(b4.groupby("system").size()) != {22}:
        raise ValueError("Require 22 longest-input attempts in every condition")
    datasets, sources, summaries = {}, {}, []
    for metric in ("mos", "sim"):
        wp, ip = run / f"{metric}_per_window.parquet", run / f"{metric}_per_item.jsonl"
        windows, items = pd.read_parquet(wp), pd.DataFrame(read_rows(ip))
        if len(items) != 650 or set(map(tuple, items[KEY].to_numpy())) != expected:
            raise ValueError("Incomplete or foreign per-item metric rows")
        if (not set(map(tuple, windows[KEY].to_numpy())) <= expected
                or windows.duplicated(KEY + ["t_center"]).any()
                or not np.isfinite(windows[metric]).all()):
            raise ValueError("Invalid, duplicated or foreign window observations")
        if sum(items.n_windows) != len(windows):
            raise ValueError("Window counts disagree with per-item records")
        # Verify means and explicit zero-window support for every attempted output.
        actual = windows.groupby(KEY)[metric].agg(["mean", "count"])
        for row in items.to_dict("records"):
            key = tuple(row[k] for k in KEY)
            count = int(row["n_windows"])
            mean = row.get("mean_" + metric)
            if count:
                if (int(actual.loc[key, "count"]) != count
                        or not np.isclose(actual.loc[key, "mean"], mean, atol=1e-7, rtol=1e-7)):
                    raise ValueError("Per-item means/counts do not reproduce the windows")
            elif mean is not None and not pd.isna(mean):
                raise ValueError("A zero-window output must not have an invented score")
        datasets[metric] = curves(windows[windows.bucket == "B4"], metric)
        sources.update({str(wp): sha(wp), str(ip): sha(ip)})
        for bucket in ("ALL", "B0", "B2", "B4"):
            cell = items if bucket == "ALL" else items[items.bucket == bucket]
            for system, group in cell.groupby("system", sort=False):
                valid = group[group.n_windows > 0]
                summaries.append(dict(system=system, bucket=bucket, metric=metric,
                    attempted=len(group), with_windows=len(valid), without_windows=len(group)-len(valid),
                    mean_over_items=None if valid.empty else valid["mean_"+metric].mean()))
    output.parent.mkdir(parents=True, exist_ok=True)
    draw(datasets["mos"], datasets["sim"], MAIN, output)
    draw(datasets["mos"], datasets["sim"], list(COLORS), output.with_name(output.name + "_all"))
    for metric, data in datasets.items():
        data.to_csv(output.with_name(output.name + "_" + metric).with_suffix(".csv"), index=False)
    pd.DataFrame(summaries).to_csv(output.with_name(output.name + "_summary").with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(json.dumps({
        "task_sha256": sha(run / "tasks.jsonl"), "sources": sources,
        "script_sha256": sha(Path(__file__)), "all_attempts_per_metric": 650,
        "figure_bucket": "B4", "attempts_per_arm": 22, "main_systems": MAIN,
        "display_labels": {system: label(system) for system in COLORS},
        "no_recorded_speech_anchor": "The published texts have no corresponding human recordings",
        "window_sec": 5, "hop_sec": 2.5, "ema_alpha": .3,
        "pointwise_mean": "one contribution per item; draw metric curves only at n >= 2",
        "missing_scores": "null with explicit support; no imputation",
        "summary": "mean of per-item means over available windows; descriptive, not all-attempt WER"
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    render(args.run.resolve(), args.output.resolve())
