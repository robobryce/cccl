#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Validate and summarize cooperative direct-output histogram sweeps.

The input JSON files are produced by ``histogram_algo_sweep.py``. This utility
checks the complete low-bin experiment grid, writes cell-level and aggregate
CSVs, records the largest regressions/wins, and renders compact throughput and
speedup figures for each histogram family.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ("even", "range", "multi_even", "multi_range")
SAMPLES = ("I32", "F64")
ELEMENTS = (1 << 20, 1 << 24, 1 << 26, 1 << 28, 1 << 30, 2_000_000_000)
BINS = (8, 16, 32, 64, 128, 256, 512)
STATIC_BINS = BINS[:-1]
SHAPES = (
    "concentrated:1.0",
    "concentrated:0.75",
    "concentrated:0.5",
    "concentrated:0.25",
    "concentrated:0.0",
    "powerlaw:0.75",
    "powerlaw:0.25",
    "hash_synonym",
    "stale_resident:0.5",
    "stale_resident:0.25",
    "temporal_phases:0.10",
    "strided_sweep",
    "sawtooth",
    "poison",
    "sawtooth:8192:2654435761:1",
)
SERIES = (
    "main",
    "smem_static",
    "smem_cooperative_static",
    "smem_dynamic",
    "smem_cooperative_dynamic",
)
NONCOOPERATIVE = {
    "smem_cooperative_static": "smem_static",
    "smem_cooperative_dynamic": "smem_dynamic",
}
LABELS = {
    "main": "main",
    "smem_static": "static",
    "smem_cooperative_static": "coop static",
    "smem_dynamic": "dynamic",
    "smem_cooperative_dynamic": "coop dynamic",
}
COLORS = {
    "main": "#e6194b",
    "smem_static": "#8c564b",
    "smem_cooperative_static": "#d62728",
    "smem_dynamic": "#1f77b4",
    "smem_cooperative_dynamic": "#2ca02c",
}
MARKERS = {
    "main": "X",
    "smem_static": "h",
    "smem_cooperative_static": "o",
    "smem_dynamic": "D",
    "smem_cooperative_dynamic": "s",
}


def parse_key(key: str) -> tuple[str, int, int, str]:
    sample, elements, bins, shape = key.split("|", 3)
    return sample, int(elements), int(bins), shape


def make_key(sample: str, elements: int, bins: int, shape: str) -> str:
    return f"{sample}|{elements}|{bins}|{shape}"


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def expected_keys(dataset: str, family: str, series: str) -> set[str]:
    bins_axis = STATIC_BINS if "static" in series else BINS
    elements_axis = ELEMENTS
    if dataset == "c32" and family.startswith("multi_") and series != "main":
        elements_axis = ELEMENTS[:4]
    return {
        make_key(sample, elements, bins, shape)
        for sample in SAMPLES
        for elements in elements_axis
        for bins in bins_axis
        for shape in SHAPES
    }


def validate_dataset(dataset: str, data: dict) -> dict:
    errors = []
    counts = {}
    for family in FAMILIES:
        if family not in data:
            errors.append(f"missing family {family}")
            continue
        counts[family] = {}
        for series in SERIES:
            cells = data[family].get(series)
            if cells is None:
                errors.append(f"{family}: missing series {series}")
                continue
            actual = set(cells)
            expected = expected_keys(dataset, family, series)
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            nonpositive = sorted(key for key, value in cells.items() if value <= 0)
            counts[family][series] = len(actual)
            if missing:
                errors.append(
                    f"{family}/{series}: {len(missing)} missing cells; first={missing[:3]}"
                )
            if unexpected:
                errors.append(
                    f"{family}/{series}: {len(unexpected)} unexpected cells; first={unexpected[:3]}"
                )
            if nonpositive:
                errors.append(
                    f"{family}/{series}: {len(nonpositive)} nonpositive cells; first={nonpositive[:3]}"
                )
    return {"ok": not errors, "errors": errors, "counts": counts}


def size_bucket(elements: int) -> str:
    if elements <= 1 << 24:
        return "small"
    if elements >= 1 << 28:
        return "large"
    return "middle"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(datasets: dict[str, dict]) -> list[dict]:
    rows = []
    for dataset, data in datasets.items():
        for family in FAMILIES:
            main = data[family]["main"]
            for series in SERIES:
                noncooperative = NONCOOPERATIVE.get(series)
                for key, throughput in data[family][series].items():
                    sample, elements, bins, shape = parse_key(key)
                    main_value = main.get(key)
                    noncooperative_value = (
                        data[family][noncooperative].get(key)
                        if noncooperative is not None
                        else None
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "series": series,
                            "sample": sample,
                            "elements": elements,
                            "size_bucket": size_bucket(elements),
                            "bins": bins,
                            "shape": shape,
                            "gib_per_s": throughput,
                            "speedup_vs_main": (
                                throughput / main_value if main_value else ""
                            ),
                            "speedup_vs_noncooperative": (
                                throughput / noncooperative_value
                                if noncooperative_value
                                else ""
                            ),
                        }
                    )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["series"] == "main" or row["size_bucket"] == "middle":
            continue
        grouped[
            (
                row["dataset"],
                row["family"],
                row["size_bucket"],
                row["series"],
            )
        ].append(row)

    output = []
    for (dataset, family, bucket, series), group in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "family": family,
                "size_bucket": bucket,
                "series": series,
                "cells": len(group),
                "gmean_gib_per_s": geometric_mean(
                    [float(row["gib_per_s"]) for row in group]
                ),
                "gmean_speedup_vs_main": geometric_mean(
                    [float(row["speedup_vs_main"]) for row in group]
                ),
                "gmean_speedup_vs_noncooperative": geometric_mean(
                    [
                        float(row["speedup_vs_noncooperative"])
                        for row in group
                        if row["speedup_vs_noncooperative"] != ""
                    ]
                ),
            }
        )
    return output


def outlier_rows(rows: list[dict], count: int = 10) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for family in FAMILIES:
            for series in SERIES[1:]:
                candidates = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["family"] == family
                    and row["series"] == series
                ]
                for comparison in (
                    "speedup_vs_main",
                    "speedup_vs_noncooperative",
                ):
                    ranked = sorted(
                        (row for row in candidates if row[comparison] != ""),
                        key=lambda row: float(row[comparison]),
                    )
                    for rank, row in enumerate(ranked[:count], start=1):
                        output.append(
                            {
                                "dataset": dataset,
                                "family": family,
                                "series": series,
                                "comparison": comparison,
                                "rank": rank,
                                "ratio": row[comparison],
                                "sample": row["sample"],
                                "elements": row["elements"],
                                "bins": row["bins"],
                                "shape": row["shape"],
                            }
                        )
    return output


def grouped_gmean(
    rows: list[dict], dataset: str, family: str, series: str, bucket: str, metric: str
) -> list[float]:
    values = defaultdict(list)
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["family"] == family
            and row["series"] == series
            and row["size_bucket"] == bucket
            and row[metric] != ""
        ):
            values[int(row["bins"])].append(float(row[metric]))
    return [geometric_mean(values[bins]) for bins in BINS]


def plot_family(rows: list[dict], dataset: str, family: str, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    metrics = (
        ("gib_per_s", "Throughput (GiB/s)"),
        ("speedup_vs_main", "Speedup vs main"),
        ("speedup_vs_noncooperative", "Cooperative / non-cooperative"),
    )
    for column, bucket in enumerate(("small", "large")):
        for row_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index][column]
            plot_series = (
                SERIES if metric != "speedup_vs_noncooperative" else SERIES[2::2]
            )
            for series in plot_series:
                values = grouped_gmean(rows, dataset, family, series, bucket, metric)
                if all(math.isnan(value) for value in values):
                    continue
                axis.plot(
                    BINS,
                    values,
                    color=COLORS[series],
                    marker=MARKERS[series],
                    label=LABELS[series],
                    linewidth=1.5,
                )
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(ylabel)
            if row_index > 0:
                axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
            if row_index == 0:
                axis.set_title(f"{bucket} workloads")
            if row_index == 2:
                axis.set_xlabel("Bins")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(f"{dataset}: {family.replace('_', '-')} geometric means", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(path: Path, aggregates: list[dict], validations: dict) -> None:
    by_key = {
        (
            row["dataset"],
            row["family"],
            row["size_bucket"],
            row["series"],
        ): row
        for row in aggregates
    }
    lines = [
        "# Cooperative direct-output histogram analysis",
        "",
        "Ratios are geometric means. Small workloads contain 1 Mi and 16 Mi elements; "
        "large workloads contain 256 Mi, 1,073,741,824, and 2,000,000,000 elements. "
        "The 64 Mi size is retained in the raw CSV but excluded from these two buckets.",
        "",
        "## Artifact validation",
        "",
    ]
    for dataset, validation in validations.items():
        lines.append(f"- `{dataset}`: {'PASS' if validation['ok'] else 'FAIL'}")
        for error in validation["errors"]:
            lines.append(f"  - {error}")
    lines.extend(["", "## Geometric-mean speedups", ""])
    for dataset in validations:
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Family | Size | CST/SST | CDY/SDY | SST/main | CST/main | SDY/main | CDY/main |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for family in FAMILIES:
            for bucket in ("small", "large"):
                values = {}
                for series in SERIES[1:]:
                    row = by_key[(dataset, family, bucket, series)]
                    values[series] = row
                lines.append(
                    "| {} | {} | {:.3f}× | {:.3f}× | {:.3f}× | {:.3f}× | {:.3f}× | {:.3f}× |".format(
                        family.replace("_", "-"),
                        bucket,
                        values["smem_cooperative_static"][
                            "gmean_speedup_vs_noncooperative"
                        ],
                        values["smem_cooperative_dynamic"][
                            "gmean_speedup_vs_noncooperative"
                        ],
                        values["smem_static"]["gmean_speedup_vs_main"],
                        values["smem_cooperative_static"]["gmean_speedup_vs_main"],
                        values["smem_dynamic"]["gmean_speedup_vs_main"],
                        values["smem_cooperative_dynamic"]["gmean_speedup_vs_main"],
                    )
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c32", type=Path, required=True)
    parser.add_argument("--u64", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    datasets = {
        "c32": json.loads(args.c32.read_text(encoding="utf-8")),
        "u64": json.loads(args.u64.read_text(encoding="utf-8")),
    }
    validations = {
        dataset: validate_dataset(dataset, data) for dataset, data in datasets.items()
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "figures").mkdir(exist_ok=True)
    (args.outdir / "artifact_validation.json").write_text(
        json.dumps(validations, indent=2) + "\n", encoding="utf-8"
    )
    if not all(validation["ok"] for validation in validations.values()):
        raise SystemExit("artifact validation failed; see artifact_validation.json")

    rows = collect_rows(datasets)
    aggregates = aggregate_rows(rows)
    outliers = outlier_rows(rows)
    write_csv(args.outdir / "measurements.csv", list(rows[0]), rows)
    write_csv(args.outdir / "geometric_means.csv", list(aggregates[0]), aggregates)
    write_csv(args.outdir / "outliers.csv", list(outliers[0]), outliers)
    write_summary(args.outdir / "analysis_summary.md", aggregates, validations)
    for dataset in datasets:
        for family in FAMILIES:
            plot_family(
                rows,
                dataset,
                family,
                args.outdir / "figures" / f"{dataset}_{family}.png",
            )
    print(
        f"validated {len(rows)} measurements; wrote {len(aggregates)} aggregates, "
        f"{len(outliers)} outliers, and {len(datasets) * len(FAMILIES)} figures"
    )


if __name__ == "__main__":
    main()
