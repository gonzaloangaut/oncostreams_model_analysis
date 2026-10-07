from __future__ import annotations

import argparse
import re
from pathlib import Path

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

# Motion columns that should be present in every file
MOTION_COLUMNS = [
    "mean_step_displacement",
    "mean_squared_step_displacement",
    "p95_step_displacement",
    "max_step_displacement",
    "msd_t0",
]


# Create a function to extract the information
def parse_metadata(
    filepath: Path,
    delta_t_fallback: float | None = None,
) -> dict:
    """Extract simulation metadata encoded in an output filename."""
    filename = filepath.name

    # Get the step
    step_match = STEP_PATTERN.search(filename)

    if step_match is None:
        raise ValueError(f"Could not read step from filename: {filename}")

    step = int(step_match.group(1))
    name_without_step = STEP_PATTERN.sub("", filename)

    # Search every field in the filename
    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_PATTERN.finditer(name_without_step)
    }

    # Seed
    seed = int(fields["rng_seed"])

    # Number of cells
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

    # Density
    requested_rho = fields.get(
        "requested_density",
        fields.get("target_density"),
    )

    actual_rho = fields.get("density")

    if requested_rho is None:
        requested_rho = actual_rho

    if actual_rho is None:
        actual_rho = requested_rho

    # Initial elongated fraction
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


# Process one motion file
def process_motion_file(
    filepath: Path,
    delta_t_fallback: float | None = None,
) -> dict:
    """
    Read one motion snapshot and return one row containing
    simulation metadata and motion observables.
    """

    # Get metadata from the filename
    metadata = parse_metadata(
        filepath,
        delta_t_fallback=delta_t_fallback,
    )

    data = pd.read_csv(
        filepath,
        skipinitialspace=True,
    )

    # Every file should contain exactly one snapshot
    if len(data) != 1:
        raise ValueError(
            f"Expected exactly one row in {filepath}, "
            f"found {len(data)}."
        )

    # Check expected columns
    missing_columns = [
        column
        for column in MOTION_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {filepath}: {missing_columns}"
        )

    observables = (
        data.iloc[0][MOTION_COLUMNS]
        .to_dict()
    )

    return {
        **metadata,
        **observables,
    }


# Process all files
def process_motion_files(
    motion_files: list[Path],
    delta_t_fallback: float | None = None,
) -> pd.DataFrame:
    """Combine all motion snapshots into one DataFrame."""

    rows = []

    for index, filepath in enumerate(motion_files, start=1):

        rows.append(
            process_motion_file(
                filepath,
                delta_t_fallback=delta_t_fallback,
            )
        )

        if index % 1000 == 0 or index == len(motion_files):
            print(
                f"Motion files processed: "
                f"{index}/{len(motion_files)}"
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# Sanity checks
def validate_motion(data: pd.DataFrame) -> None:
    """Perform basic checks on the processed motion observables."""

    duplicated = data.duplicated(
        subset=SNAPSHOT_KEYS,
    ).sum()

    negative_columns = {
        column: int((data[column] < 0).sum())
        for column in MOTION_COLUMNS
    }

    print("\nSanity checks:")
    print(
        "Duplicated simulation-snapshot rows:",
        duplicated,
    )

    for column, number_negative in negative_columns.items():
        print(
            f"Negative values in {column}:",
            number_negative,
        )


# Main
def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Consolidate motion output files "
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
            "<data_root>/processed/motion"
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
        else data_root / "processed" / "motion"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find all motion files
    motion_files = sorted(
        data_root.rglob("motion_culture_*.dat")
    )

    print(
        f"Motion files found: {len(motion_files)}"
    )

    if not motion_files:
        raise FileNotFoundError(
            f"No motion files found below {data_root}"
        )

    # Process data
    motion = process_motion_files(
        motion_files,
        delta_t_fallback=args.delta_t,
    )

    # Sort data
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

    motion = (
        motion
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    # Validate data
    validate_motion(
        motion
    )

    # Save
    output_path = (
        output_dir
        / "motion.parquet"
    )

    motion.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nProcessed motion data written to:"
        f"\n{output_path}"
    )

    print(
        f"\nRows written: {len(motion)}"
    )


if __name__ == "__main__":
    main()
