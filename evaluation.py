"""
evaluation.py - Summarise all test results into one table.

Place this file under:
    /data/home/hky/DULRTC/hky_try_3/evaluation.py

It searches for:
    metrics_01.txt
    metrics_01.json

Typical test output:
    runs/mask15_safe_bs1/test/Mask15/metrics_01.txt
    runs/mask10_xxx/test/Mask10/metrics_01.txt
    runs/fiber15_xxx/test/Fiber15/metrics_01.txt

Usage:
    cd /data/home/hky/DULRTC/hky_try_3

    python evaluation.py --root test

Outputs:
    eval/
        evaluation_summary.txt
        evaluation_summary.csv
        evaluation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRIC_NAMES = ["PSNR", "RMSE", "NMSE", "outage_mse", "outage_rmse"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="runs",
        help="Root directory to recursively search for metrics_01.txt/json.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval/evaluation_summary",
        help="Output prefix. Generates .txt, .csv and .md.",
    )
    parser.add_argument(
        "--prefer-json",
        action="store_true",
        help="Prefer metrics_01.json over metrics_01.txt when both exist.",
    )
    return parser.parse_args()


def normalise_exp_name(name: str) -> Tuple[str, int]:
    m = re.search(r"(mask|fiber)[_\-\s]*(\d+)", name.strip(), flags=re.IGNORECASE)
    if not m:
        return name.strip(), 999999

    kind = m.group(1).lower()
    num = int(m.group(2))
    kind = "Mask" if kind == "mask" else "Fiber"
    return kind, num


def display_exp_name(name: str) -> str:
    kind, num = normalise_exp_name(name)
    if kind in ("Mask", "Fiber") and num != 999999:
        return f"{kind} {num}"
    return name


def exp_sort_key(row: Dict[str, str]):
    exp = row["Experiment"]
    kind, num = normalise_exp_name(exp)

    if kind == "Mask":
        kind_order = 0
    elif kind == "Fiber":
        kind_order = 1
    else:
        kind_order = 9

    return (kind_order, -num, exp)


def parse_metrics_txt(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    avg_line = None
    for line in lines:
        if line.strip().startswith("[average]"):
            avg_line = line.strip()
            break

    if avg_line is None:
        return None

    parts = avg_line.split()

    # [average] PSNR RMSE NMSE outage_mse outage_rmse
    if len(parts) < 6:
        return None

    try:
        return {
            "PSNR": float(parts[1]),
            "RMSE": float(parts[2]),
            "NMSE": float(parts[3]),
            "outage_mse": float(parts[4]),
            "outage_rmse": float(parts[5]),
        }
    except ValueError:
        return None


def parse_metrics_json(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Current metrics_01.json style
    if "average" in data and isinstance(data["average"], dict):
        avg = data["average"]
        out = {}
        for k in METRIC_NAMES:
            if k in avg:
                out[k] = float(avg[k])
        return out if out else None

    # Compatibility with older metrics.json style
    key_map = {
        "PSNR": "PSNR_dB",
        "RMSE": "RMSE",
        "NMSE": "NMSE_dB",
        "outage_mse": "OUT_RATE",
    }

    out = {}
    for new_k, old_k in key_map.items():
        if old_k in data and isinstance(data[old_k], dict) and "mean" in data[old_k]:
            out[new_k] = float(data[old_k]["mean"])

    return out if out else None


def infer_experiment_name(metrics_file: Path) -> str:
    parent = metrics_file.parent.name

    if re.search(r"(mask|fiber)[_\-\s]*\d+", parent, flags=re.IGNORECASE):
        return parent

    return parent


def discover_metric_files(root: Path, prefer_json: bool = False) -> List[Path]:
    txt_files = list(root.rglob("metrics_01.txt"))
    json_files = list(root.rglob("metrics_01.json"))

    by_dir = {}

    for path in txt_files:
        by_dir.setdefault(path.parent, {})["txt"] = path

    for path in json_files:
        by_dir.setdefault(path.parent, {})["json"] = path

    selected = []

    for _, files in by_dir.items():
        if prefer_json and "json" in files:
            selected.append(files["json"])
        elif "txt" in files:
            selected.append(files["txt"])
        elif "json" in files:
            selected.append(files["json"])

    return sorted(selected)


def collect_results(root: Path, prefer_json: bool = False) -> List[Dict[str, str]]:
    metric_files = discover_metric_files(root, prefer_json=prefer_json)

    rows: List[Dict[str, str]] = []

    for path in metric_files:
        if path.suffix == ".json":
            metrics = parse_metrics_json(path)
        else:
            metrics = parse_metrics_txt(path)

        if metrics is None:
            print(f"[warn] failed to parse: {path}")
            continue

        exp_raw = infer_experiment_name(path)
        exp = display_exp_name(exp_raw)

        row = {
            "Experiment": exp,
            "Source": str(path),
        }

        for k in METRIC_NAMES:
            if k in metrics:
                row[k] = f"{metrics[k]:.4f}"
            else:
                row[k] = ""

        rows.append(row)

    rows.sort(key=exp_sort_key)
    return rows


def make_text_table(rows: List[Dict[str, str]]) -> str:
    headers = ["Experiment"] + METRIC_NAMES

    if not rows:
        return "No metrics found.\n"

    col_widths = []

    for h in headers:
        max_len = len(h)
        for row in rows:
            max_len = max(max_len, len(str(row.get(h, ""))))
        col_widths.append(max_len + 2)

    def fmt(values):
        return "".join(str(values[i]).ljust(col_widths[i]) for i in range(len(values)))

    lines = []
    lines.append(fmt(headers))
    lines.append(fmt(["-" * (w - 2) for w in col_widths]))

    for row in rows:
        lines.append(fmt([row.get(h, "") for h in headers]))

    return "\n".join(lines) + "\n"


def make_markdown_table(rows: List[Dict[str, str]]) -> str:
    headers = ["Experiment"] + METRIC_NAMES

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values = [row.get(h, "") for h in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def save_csv(rows: List[Dict[str, str]], out_path: Path):
    headers = ["Experiment"] + METRIC_NAMES + ["Source"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    rows = collect_results(root, prefer_json=args.prefer_json)

    out_prefix = Path(args.out)

    txt_path = out_prefix.with_suffix(".txt")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    txt = make_text_table(rows)
    md = make_markdown_table(rows)

    txt_path.write_text(txt, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    save_csv(rows, csv_path)

    print("\n=== Evaluation Summary ===")
    print(txt)

    print(f"Saved TXT: {txt_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD : {md_path}")

    if not rows:
        print("[warn] No valid metrics_01.txt/json files found.")
        print("[hint] Expected files like:")
        print("       runs/.../test/Mask15/metrics_01.txt")
        print("       runs/.../test/Fiber15/metrics_01.txt")


if __name__ == "__main__":
    main()
