"""
evaluation.py - Summarise test results and optionally aggregate over seeds.

Typical usage:

Single seed:
    python evaluation.py --root test_seed0 --out eval/evaluation_seed0

Multiple seeds:
    python evaluation.py \
        --roots test_seed0 test_seed42 test_seed2024 \
        --out eval/evaluation_summary

Outputs:
    eval/evaluation_summary.txt
    eval/evaluation_summary.csv
    eval/evaluation_summary.md

If multiple roots are given, the script reports mean ± std over seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


METRIC_NAMES = ["PSNR", "RMSE", "NMSE", "outage_mse", "outage_rmse"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Single root directory to recursively search for metrics_01.txt/json.",
    )

    parser.add_argument(
        "--roots",
        type=str,
        nargs="+",
        default=None,
        help="Multiple root directories, e.g., test_seed0 test_seed42 test_seed2024.",
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


def exp_sort_key_from_name(exp: str):
    kind, num = normalise_exp_name(exp)

    if kind == "Mask":
        kind_order = 0
    elif kind == "Fiber":
        kind_order = 1
    else:
        kind_order = 9

    return kind_order, -num, exp


def exp_sort_key(row: Dict[str, str]):
    return exp_sort_key_from_name(row["Experiment"])


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

    if "average" in data and isinstance(data["average"], dict):
        avg = data["average"]
        out = {}
        for k in METRIC_NAMES:
            if k in avg:
                out[k] = float(avg[k])
        return out if out else None

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


def collect_results_one_root(
    root: Path,
    prefer_json: bool = False,
    seed_name: str = "",
) -> List[Dict[str, str]]:
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
            "Seed": seed_name,
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


def infer_seed_name(root: Path) -> str:
    name = root.name
    m = re.search(r"seed(\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return name


def aggregate_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}

    for row in rows:
        grouped.setdefault(row["Experiment"], []).append(row)

    agg_rows: List[Dict[str, str]] = []

    for exp, exp_rows in grouped.items():
        out = {
            "Experiment": exp,
            "N": str(len(exp_rows)),
        }

        for metric in METRIC_NAMES:
            vals = []
            for row in exp_rows:
                if row.get(metric, "") != "":
                    vals.append(float(row[metric]))

            if not vals:
                out[metric] = ""
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                continue

            arr = np.asarray(vals, dtype=float)
            mean = arr.mean()
            std = arr.std(ddof=1) if len(arr) >= 2 else 0.0

            out[metric] = f"{mean:.4f} ± {std:.4f}"
            out[f"{metric}_mean"] = f"{mean:.4f}"
            out[f"{metric}_std"] = f"{std:.4f}"

        agg_rows.append(out)

    agg_rows.sort(key=exp_sort_key)
    return agg_rows


def make_text_table(rows: List[Dict[str, str]], aggregate: bool = False) -> str:
    headers = ["Experiment"]

    if aggregate:
        headers += ["N"] + METRIC_NAMES
    else:
        headers += ["Seed"] + METRIC_NAMES

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


def make_markdown_table(rows: List[Dict[str, str]], aggregate: bool = False) -> str:
    headers = ["Experiment"]

    if aggregate:
        headers += ["N"] + METRIC_NAMES
    else:
        headers += ["Seed"] + METRIC_NAMES

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values = [row.get(h, "") for h in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def save_csv(rows: List[Dict[str, str]], out_path: Path, aggregate: bool = False):
    if aggregate:
        headers = ["Experiment", "N"]
        for metric in METRIC_NAMES:
            headers += [metric, f"{metric}_mean", f"{metric}_std"]
    else:
        headers = ["Seed", "Experiment"] + METRIC_NAMES + ["Source"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    if args.roots is not None:
        root_paths = [Path(x) for x in args.roots]
        aggregate = True
    elif args.root is not None:
        root_paths = [Path(args.root)]
        aggregate = False
    else:
        raise ValueError("Please provide either --root or --roots.")

    all_rows: List[Dict[str, str]] = []

    for root in root_paths:
        if not root.exists():
            print(f"[warn] root directory not found, skipped: {root}")
            continue

        seed_name = infer_seed_name(root)
        rows = collect_results_one_root(
            root,
            prefer_json=args.prefer_json,
            seed_name=seed_name,
        )
        all_rows.extend(rows)

    if aggregate:
        rows_to_save = aggregate_rows(all_rows)
    else:
        rows_to_save = all_rows

    out_prefix = Path(args.out)

    txt_path = out_prefix.with_suffix(".txt")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    txt = make_text_table(rows_to_save, aggregate=aggregate)
    md = make_markdown_table(rows_to_save, aggregate=aggregate)

    txt_path.write_text(txt, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    save_csv(rows_to_save, csv_path, aggregate=aggregate)

    print("\n=== Evaluation Summary ===")
    print(txt)

    print(f"Saved TXT: {txt_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD : {md_path}")

    if not rows_to_save:
        print("[warn] No valid metrics_01.txt/json files found.")
        print("[hint] Expected files like:")
        print("       test_seed0/Mask15/metrics_01.txt")
        print("       test_seed42/Fiber15/metrics_01.txt")


if __name__ == "__main__":
    main()