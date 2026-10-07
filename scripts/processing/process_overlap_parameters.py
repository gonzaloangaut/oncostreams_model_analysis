from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Read the information encoded in the file names
KNOWN_FIELDS = [
    "initial_number_of_cells",
    "initial_fraction_elongated",
    "requested_density",
    "target_density",
    "density",
    "requested_nc",
    "reference_nc",
    "initial_nc",
    "target_nc",
    "removed_nc",
    "initial_f_e",
    "force",
    "rng_seed",
]


FIELD_PATTERN = re.compile(
    r"_(?P<key>" + "|".join(sorted(KNOWN_FIELDS, key=len, reverse=True)) + r")="
    r"(?P<value>.*?)"
    r"(?=_(?:" + "|".join(sorted(KNOWN_FIELDS, key=len, reverse=True)) + r")=|$)"
)


STEP_PATTERN = re.compile(r"_step=(\d+)\.dat$")

DELTA_T_DIR_PATTERN = re.compile(
    r"^(?:delta_t_)?0_(?P<fraction>\d+)(?:_\d+)?$"
)


def resolve_delta_t(
    filepath: Path,
    fallback: float | None = None,
) -> float:
    """Infer delta_t from a parent directory such as 0_025_1.

    If no such directory is present, ``fallback`` can be supplied through
    the command-line ``--delta-t`` option.
    """
    for part in reversed(filepath.parent.parts):
        match = DELTA_T_DIR_PATTERN.fullmatch(part)
        if match is not None:
            return float(f"0.{match.group('fraction')}")

    if fallback is not None:
        if fallback <= 0:
            raise ValueError("delta_t must be positive.")
        return float(fallback)

    raise ValueError(
        "Could not infer delta_t from the directory tree for "
        f"{filepath}. Expected a directory such as 0_1, 0_05, "
        "0_025_1, or 0_01_2. Alternatively pass --delta-t."
    )


# Columns identifying one simulation snapshot within this parameter sweep
SNAPSHOT_KEYS = [
    "N",
    "reference_N",
    "actual_N",
    "rho",
    "actual_rho",
    "initial_fraction_elongated",
    "force",
    "seed",
    "delta_t",
    "step",
]


def parse_force_parameters(force_name: str) -> dict:
    """Extract numeric parameters without modifying the full force name."""
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    parameters = {}

    for column, token in (
        ("kappa", "k"),
        ("lambda_core", "lambda_core"),
    ):
        match = re.search(
            rf"(?:^|_){token}=({number_pattern})(?=_|$)",
            force_name,
        )

        # Missing parameters remain unknown rather than assuming a default
        parameters[column] = (
            float(match.group(1))
            if match is not None
            else float("nan")
        )

    return parameters

OVERLAP_COLUMNS = [
    "tic_start",
    "tic_end",
    "number_of_steps",
    "max_normalized_overlap",
    "mean_normalized_overlap",
]


def parse_metadata(
    filepath: Path,
    delta_t_fallback: float | None = None,
) -> dict:
    """Extract simulation metadata encoded in an output filename."""
    filename = filepath.name

    step_match = STEP_PATTERN.search(filename)

    if step_match is None:
        raise ValueError(f"Could not read step from filename: {filename}")

    step = int(step_match.group(1))
    name_without_step = STEP_PATTERN.sub("", filename)

    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_PATTERN.finditer(name_without_step)
    }

    seed = int(fields["rng_seed"])

    requested_n = fields.get(
        "requested_nc",
        fields.get("target_nc"),
    )

    initial_n = fields.get(
        "initial_nc",
        fields.get("initial_number_of_cells"),
    )

    reference_n = fields.get("reference_nc")

    if requested_n is None:
        requested_n = initial_n

    if initial_n is None:
        initial_n = requested_n

    if reference_n is None:
        reference_n = requested_n

    requested_rho = fields.get(
        "requested_density",
        fields.get("target_density"),
    )

    actual_rho = fields.get("density")

    if requested_rho is None:
        requested_rho = actual_rho

    if actual_rho is None:
        actual_rho = requested_rho

    initial_fraction_elongated = fields.get(
        "initial_f_e",
        fields.get("initial_fraction_elongated", "0"),
    )

    if requested_n is None or requested_rho is None:
        raise ValueError(
            "Could not identify N and density from filename: "
            f"{filename}"
        )

    force_name = fields.get("force", "")
    delta_t = resolve_delta_t(filepath, fallback=delta_t_fallback)
    return {
        "N": int(requested_n),
        "reference_N": int(reference_n),
        "actual_N": int(initial_n),
        "rho": float(requested_rho),
        "actual_rho": float(actual_rho),
        "seed": seed,
        "step": step,
        "delta_t": delta_t,
        "time": step * delta_t,
        "initial_fraction_elongated": float(initial_fraction_elongated),
        "force": force_name,
        **parse_force_parameters(force_name),
    }


def process_overlap_file(
    filepath: Path,
    delta_t_fallback: float | None = None,
) -> dict:
    """Read one overlap interval and return one processed row."""

    metadata = parse_metadata(
        filepath,
        delta_t_fallback=delta_t_fallback,
    )

    data = pd.read_csv(
        filepath,
        skipinitialspace=True,
    )

    if len(data) != 1:
        raise ValueError(
            f"Expected exactly one row in {filepath}, "
            f"found {len(data)}."
        )

    missing_columns = [
        column
        for column in OVERLAP_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {filepath}: {missing_columns}"
        )

    row = data.iloc[0]

    return {
        **metadata,
        "tic_start": int(row["tic_start"]),
        "tic_end": int(row["tic_end"]),
        "number_of_steps": int(row["number_of_steps"]),
        "max_normalized_overlap": float(
            row["max_normalized_overlap"]
        ),
        "mean_normalized_overlap": float(
            row["mean_normalized_overlap"]
        ),
        "time_start": int(row["tic_start"]) * metadata["delta_t"],
        "time_end": int(row["tic_end"]) * metadata["delta_t"],
        "interval_duration": (
            int(row["number_of_steps"]) * metadata["delta_t"]
        ),
    }


def process_overlap_files(
    overlap_files: list[Path],
    delta_t_fallback: float | None = None,
) -> pd.DataFrame:
    """Combine all overlap intervals into one DataFrame."""

    rows = []

    for index, filepath in enumerate(overlap_files, start=1):
        rows.append(
            process_overlap_file(
                filepath,
                delta_t_fallback=delta_t_fallback,
            )
        )

        if index % 1000 == 0 or index == len(overlap_files):
            print(
                f"Overlap files processed: "
                f"{index}/{len(overlap_files)}"
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def validate_overlap(data: pd.DataFrame) -> None:
    """Perform consistency checks on the overlap intervals."""

    duplicated = data.duplicated(
        subset=SNAPSHOT_KEYS,
    ).sum()

    step_mismatches = int(
        (data["step"] != data["tic_end"]).sum()
    )

    expected_number_of_steps = (
        data["tic_end"]
        - data["tic_start"]
        + 1
    )

    interval_length_mismatches = int(
        (
            data["number_of_steps"]
            != expected_number_of_steps
        ).sum()
    )

    negative_overlaps = int(
        (data["max_normalized_overlap"] < 0).sum()
    )

    overlaps_above_one = int(
        (data["max_normalized_overlap"] > 1 + 1e-12).sum()
    )

    nonfinite_overlaps = int(
        (~np.isfinite(data["max_normalized_overlap"])).sum()
    )

    negative_mean_overlaps = int(
        (data["mean_normalized_overlap"] < 0).sum()
    )

    mean_overlaps_above_one = int(
        (data["mean_normalized_overlap"] > 1 + 1e-12).sum()
    )

    nonfinite_mean_overlaps = int(
        (~np.isfinite(data["mean_normalized_overlap"])).sum()
    )

    mean_above_max = int(
        (
            data["mean_normalized_overlap"]
            > data["max_normalized_overlap"] + 1e-12
        ).sum()
    )

    print("\nSanity checks:")
    print(
        "Duplicated simulation-snapshot rows:",
        duplicated,
    )
    print(
        "Rows where filename step differs from tic_end:",
        step_mismatches,
    )
    print(
        "Rows with an inconsistent interval length:",
        interval_length_mismatches,
    )
    print(
        "Negative maximum normalized overlaps:",
        negative_overlaps,
    )
    print(
        "Maximum normalized overlaps above one:",
        overlaps_above_one,
    )
    print(
        "Non-finite maximum normalized overlaps:",
        nonfinite_overlaps,
    )
    print(
        "Negative mean normalized overlaps:",
        negative_mean_overlaps,
    )
    print(
        "Mean normalized overlaps above one:",
        mean_overlaps_above_one,
    )
    print(
        "Non-finite mean normalized overlaps:",
        nonfinite_mean_overlaps,
    )
    print(
        "Rows where mean overlap exceeds maximum overlap:",
        mean_above_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate overlap-parameter output files "
            "into one parquet dataset."
        )
    )

    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing simulation outputs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<data_root>/processed/overlap"
        ),
    )

    parser.add_argument(
        "--delta-t",
        type=float,
        default=None,
        help=(
            "Fallback integration timestep when it cannot be inferred "
            "from a parent directory such as 0_05 or 0_025_1."
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "processed" / "overlap"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlap_files = sorted(
        data_root.rglob("overlap_parameters_*.dat")
    )

    print(
        f"Overlap files found: {len(overlap_files)}"
    )

    if not overlap_files:
        raise FileNotFoundError(
            f"No overlap files found below {data_root}"
        )

    overlap = process_overlap_files(
        overlap_files,
        delta_t_fallback=args.delta_t,
    )

    sort_columns = [
        "N",
        "lambda_core",
        "kappa",
        "delta_t",
        "rho",
        "force",
        "initial_fraction_elongated",
        "seed",
        "step",
    ]

    overlap = (
        overlap
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    validate_overlap(
        overlap
    )

    output_path = (
        output_dir
        / "overlap.parquet"
    )

    overlap.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nProcessed overlap data written to:"
        f"\n{output_path}"
    )

    print(
        f"\nRows written: {len(overlap)}"
    )


if __name__ == "__main__":
    main()
