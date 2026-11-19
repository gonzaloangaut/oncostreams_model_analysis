#!/usr/bin/env python3
"""Consolidate local-order summaries and occupied-box grids from Oncostreams.

This processor targets the CSV-like .dat output of
`DatOutput_local_order_parameters` in oncostreams 0.0.2. It does not
modify or remove any simulation output files.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "initial_number_of_cells", "initial_fraction_elongated",
    "requested_density", "target_density", "density",
    "requested_nc", "reference_nc", "initial_nc", "target_nc",
    "removed_nc", "initial_f_e", "force", "rng_seed",
]
KEYS = "|".join(sorted(FIELDS, key=len, reverse=True))
FIELD_RE = re.compile(rf"_(?P<key>{KEYS})=(?P<value>.*?)(?=_(?:{KEYS})=|$)")
STEP_RE = re.compile(r"_step=(\d+)\.dat$")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

SUMMARY_COLUMNS = {
    "number_of_bins", "actual_box_size", "total_number_of_cells",
    "number_elongated", "fraction_elongated", "number_occupied_boxes",
    "number_boxes_with_elongated", "number_boxes_with_at_least_2_elongated",
    "mean_polar_order", "mean_nematic_order", "mean_polar_order_min_2",
    "mean_nematic_order_min_2", "weighted_mean_polar_order",
    "weighted_mean_nematic_order", "mean_polar_order_hat",
    "mean_nematic_order_hat",
}
GRID_COLUMNS = {
    "grid_x", "grid_y", "center_x", "center_y", "occupancy",
    "number_round", "number_elongated", "fraction_elongated",
    "polar_order", "nematic_order", "polar_order_hat", "nematic_order_hat",
}
GRID_HEADER_KEYS = {"side", "number_of_bins", "target_box_size", "actual_box_size"}


def metadata_from_name(path: Path, delta_t: float) -> dict:
    match = STEP_RE.search(path.name)
    if match is None:
        raise ValueError(f"Missing _step=NNNNN.dat in {path}")
    step = int(match.group(1))
    base = STEP_RE.sub("", path.name)
    fields = {m.group("key"): m.group("value") for m in FIELD_RE.finditer(base)}
    try:
        n = int(fields.get("requested_nc") or fields.get("target_nc")
                or fields.get("initial_nc") or fields["initial_number_of_cells"])
        initial_n = int(fields.get("initial_nc") or fields.get("initial_number_of_cells") or n)
        rho = float(fields.get("requested_density") or fields.get("target_density")
                    or fields["density"])
        actual_rho = float(fields.get("density") or rho)
        seed = int(fields["rng_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot decode N, density or seed: {path}") from exc
    force = fields.get("force", "")
    k_match = re.search(rf"(?:^|_)k=({NUMBER})(?=_|$)", force)
    l_match = re.search(rf"(?:^|_)lambda_core=({NUMBER})(?=_|$)", force)
    if k_match is None or l_match is None:
        raise ValueError(f"Cannot decode kappa or lambda_core: {path}")
    return {
        "N": n,
        "reference_N": int(fields.get("reference_nc") or n),
        "actual_N": initial_n,
        "rho": rho,
        "actual_rho": actual_rho,
        "seed": seed,
        "step": step,
        "delta_t": delta_t,
        "time": step * delta_t,
        "initial_fraction_elongated": float(fields.get("initial_f_e")
                                            or fields.get("initial_fraction_elongated")
                                            or 0.0),
        "force": force,
        "kappa": float(k_match.group(1)),
        "lambda_core": float(l_match.group(1)),
    }


def header_from_grid(file_handle, path: Path) -> dict:
    line = file_handle.readline().strip()
    if not line.startswith("# "):
        raise ValueError(f"Invalid grid header in {path}: {line!r}")
    try:
        values = dict(field.strip().split("=", 1) for field in line[2:].split(","))
        if set(values) != GRID_HEADER_KEYS:
            raise ValueError(f"Unexpected grid header keys {sorted(values)}")
        return {
            "side": float(values["side"]),
            "number_of_bins": int(values["number_of_bins"]),
            "target_box_size": float(values["target_box_size"]),
            "actual_box_size": float(values["actual_box_size"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid grid header in {path}") from exc


def read_summary(path: Path, delta_t: float) -> pd.DataFrame:
    data = pd.read_csv(path)
    if len(data) != 1 or not SUMMARY_COLUMNS.issubset(data.columns):
        raise ValueError(f"Unexpected local summary shape/columns in {path}")
    for key, val in metadata_from_name(path, delta_t).items():
        data[key] = val
    if int(data.loc[0, "total_number_of_cells"]) != int(data.loc[0, "N"]):
        raise ValueError(f"Local summary population mismatch in {path}")
    return data


def read_grid(path: Path, delta_t: float) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        header = header_from_grid(handle, path)
        data = pd.read_csv(handle)
    if data.empty or not GRID_COLUMNS.issubset(data.columns):
        raise ValueError(f"Unexpected local grid shape/columns in {path}")
    meta = metadata_from_name(path, delta_t)
    if int(data["occupancy"].sum()) != meta["N"]:
        raise ValueError(f"Local grid population mismatch in {path}")
    if not (data["occupancy"] == data["number_round"] + data["number_elongated"]).all():
        raise ValueError(f"Inconsistent box cell counts in {path}")
    if data.duplicated(["grid_x", "grid_y"]).any():
        raise ValueError(f"Duplicate box coordinates in {path}")
    if len(data) > header["number_of_bins"] ** 2:
        raise ValueError(f"Too many occupied boxes in {path}")
    for key, val in {**meta, **header}.items():
        data[key] = val
    return data


def check_identity(frame: pd.DataFrame, kind: str, expected_steps: list[int] | None) -> None:
    keys = ["rho", "kappa", "seed", "step"]
    snapshot_ids = frame[keys].drop_duplicates()
    if kind == "summary" and len(snapshot_ids) != len(frame):
        raise ValueError("Duplicate local-order summary snapshots")
    if expected_steps is not None:
        found = set(snapshot_ids["step"].astype(int).unique())
        expected = set(expected_steps)
        if found != expected:
            raise ValueError(f"{kind} time steps mismatch: {sorted(found)} vs {sorted(expected)}")
        counts = snapshot_ids.groupby(["rho", "kappa", "seed"]).size()
        if counts.empty or (counts != len(expected)).any():
            raise ValueError(f"{kind} missing snapshots in some realizations")
    if frame["kappa"].isna().any():
        raise ValueError(f"{kind} has missing kappa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--delta-t", type=float, required=True)
    parser.add_argument("--expected-realizations", type=int, default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--summary-period", type=float, default=100.0)
    parser.add_argument("--grid-period", type=float, default=500.0)
    args = parser.parse_args()
    if not np.isfinite(args.delta_t) or args.delta_t <= 0:
        parser.error("--delta-t must be positive")

    root = args.data_root.expanduser().resolve()
    output = (args.output_dir or root / "processed" / "local_order").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_files = sorted(root.rglob("local_order_summary_*.dat"))
    grid_files = sorted(root.rglob("local_order_grid_*.dat"))
    print(f"Local summary files: {len(summary_files)}", flush=True)
    print(f"Local grid files: {len(grid_files)}", flush=True)
    if not summary_files or not grid_files:
        raise FileNotFoundError("Local-order summary or grid .dat files missing")

    def steps(period: float) -> list[int] | None:
        if args.final_time is None:
            return None
        max_step = round(args.final_time / args.delta_t)
        interval = round(period / args.delta_t)
        if interval < 1 or not np.isclose(interval * args.delta_t, period, rtol=0, atol=1e-8):
            raise ValueError("Output period must be an exact multiple of delta_t")
        return sorted(set(range(0, max_step + 1, interval)) | {max_step})

    summary_expected = steps(args.summary_period)
    grid_expected = steps(args.grid_period)
    if args.expected_realizations is not None:
        for kind, paths, expected in (
            ("summary", summary_files, summary_expected),
            ("grid", grid_files, grid_expected),
        ):
            if expected is not None and len(paths) != args.expected_realizations * len(expected):
                raise ValueError(f"{kind}: expected {args.expected_realizations * len(expected)} files, got {len(paths)}")

    print("Reading summary snapshots...", flush=True)
    summaries = []
    for i, path in enumerate(summary_files, 1):
        summaries.append(read_summary(path, args.delta_t))
        if i % 5000 == 0:
            print(f"  {i}/{len(summary_files)} summaries", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.sort_values(["rho", "kappa", "seed", "step"], inplace=True, ignore_index=True)
    check_identity(summary, "summary", summary_expected)

    print("Reading full occupied-box grids...", flush=True)
    grids = []
    for i, path in enumerate(grid_files, 1):
        grids.append(read_grid(path, args.delta_t))
        if i % 1000 == 0:
            print(f"  {i}/{len(grid_files)} grids", flush=True)
    grid = pd.concat(grids, ignore_index=True)
    grid.sort_values(["rho", "kappa", "seed", "step", "grid_y", "grid_x"], inplace=True, ignore_index=True)
    check_identity(grid, "grid", grid_expected)

    if args.expected_realizations is not None:
        ids = summary[["rho", "kappa", "seed"]].drop_duplicates()
        grid_ids = grid[["rho", "kappa", "seed"]].drop_duplicates()
        if len(ids) != args.expected_realizations or len(grid_ids) != args.expected_realizations:
            raise ValueError("Unexpected number of distinct realizations")
        if not ids.merge(grid_ids, on=["rho", "kappa", "seed"], how="outer", indicator=True)["_merge"].eq("both").all():
            raise ValueError("Summary/grid realization identities differ")

    summary_out = output / "local_order_summary.parquet"
    grid_out = output / "local_order_grid.parquet"
    summary.to_parquet(summary_out, index=False)
    grid.to_parquet(grid_out, index=False)
    print(f"Wrote {summary_out}: {len(summary)} rows", flush=True)
    print(f"Wrote {grid_out}: {len(grid)} rows", flush=True)


if __name__ == "__main__":
    main()
