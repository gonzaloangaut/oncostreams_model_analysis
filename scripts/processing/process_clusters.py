from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

# Current delta-t runs are stored in directories such as
# 0_1, 0_05, 0_025_1, 0_025_2, 0_01_1, ...
DELTA_T_DIR_PATTERN = re.compile(
    r"^(?:delta_t_)?0_(?P<fraction>\d+)(?:_\d+)?$"
)

# Columns identifying one simulation snapshot.
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

# A cluster graph is defined by phenotype, alignment rule and range factor.
GRAPH_KEYS = [
    "phenotype",
    "alignment",
    "range_factor",
]


def resolve_delta_t(
    filepath: Path,
    fallback: float | None = None,
) -> float:
    """Infer delta_t from a parent directory such as 0_025_1.

    If the path does not encode delta_t, ``fallback`` can be supplied through
    the command-line ``--delta-t`` option. This keeps the processor reusable
    for future fixed-delta-t campaigns.
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


def parse_force_parameters(force_name: str) -> dict:
    """Extract numeric force parameters without modifying the full name."""
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
        parameters[column] = (
            float(match.group(1))
            if match is not None
            else float("nan")
        )

    return parameters


def parse_metadata(
    filepath: Path,
    delta_t_fallback: float | None = None,
) -> dict:
    """Extract simulation metadata encoded in an output filename/path."""
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

    requested_n = fields.get("requested_nc", fields.get("target_nc"))
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


def add_metadata(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Add simulation metadata columns to one output table."""
    data = data.copy()
    for key, value in metadata.items():
        data[key] = value

    metadata_columns = [
        "N",
        "reference_N",
        "actual_N",
        "rho",
        "actual_rho",
        "seed",
        "delta_t",
        "step",
        "time",
        "initial_fraction_elongated",
        "force",
        "kappa",
        "lambda_core",
    ]

    other_columns = [
        column
        for column in data.columns
        if column not in metadata_columns
    ]

    return data[metadata_columns + other_columns]


def calculate_snapshot_statistics(
    graph_data: pd.DataFrame,
    metadata: dict,
    phenotype: str,
    alignment: str,
    range_factor: float,
) -> dict:
    """Calculate raw-derived cluster observables for one graph snapshot."""
    sizes = graph_data["size"].to_numpy(dtype=int)
    sizes_sorted = np.sort(sizes)[::-1]

    number_of_cells_phenotype = int(np.sum(sizes))
    number_of_clusters = int(sizes.size)
    system_number_of_cells = int(metadata["actual_N"])

    s1 = int(sizes_sorted[0]) if sizes_sorted.size >= 1 else 0
    s2 = int(sizes_sorted[1]) if sizes_sorted.size >= 2 else 0

    denominator = s1 + s2
    psi = (s1 - s2) / denominator if denominator > 0 else np.nan

    number_of_singletons = int(np.sum(sizes == 1))

    if sizes_sorted.size > 1:
        finite_sizes = sizes_sorted[1:]
        arithmetic_mean_without_largest = float(np.mean(finite_sizes))
        mean_finite_cluster_size = float(
            np.sum(finite_sizes**2) / np.sum(finite_sizes)
        )
    else:
        arithmetic_mean_without_largest = np.nan
        mean_finite_cluster_size = np.nan

    if number_of_cells_phenotype > 0:
        s1_over_n_phenotype = s1 / number_of_cells_phenotype
        s2_over_n_phenotype = s2 / number_of_cells_phenotype
        isolated_fraction_phenotype = (
            number_of_singletons / number_of_cells_phenotype
        )
        cluster_density = number_of_clusters / number_of_cells_phenotype
    else:
        s1_over_n_phenotype = np.nan
        s2_over_n_phenotype = np.nan
        isolated_fraction_phenotype = np.nan
        cluster_density = np.nan

    if system_number_of_cells > 0:
        s1_over_n = s1 / system_number_of_cells
        s2_over_n = s2 / system_number_of_cells
        fraction_phenotype = (
            number_of_cells_phenotype / system_number_of_cells
        )
        isolated_fraction_total = (
            number_of_singletons / system_number_of_cells
        )
    else:
        s1_over_n = np.nan
        s2_over_n = np.nan
        fraction_phenotype = np.nan
        isolated_fraction_total = np.nan

    return {
        **metadata,
        "phenotype": phenotype,
        "alignment": alignment,
        "range_factor": float(range_factor),
        "system_number_of_cells": system_number_of_cells,
        "number_of_cells_phenotype": number_of_cells_phenotype,
        "fraction_phenotype": fraction_phenotype,
        "number_of_clusters": number_of_clusters,
        "cluster_density": cluster_density,
        "S1": s1,
        "S2": s2,
        "S1_over_N": s1_over_n,
        "S2_over_N": s2_over_n,
        "S1_over_Nphenotype": s1_over_n_phenotype,
        "S2_over_Nphenotype": s2_over_n_phenotype,
        "psi": psi,
        "number_of_singletons": number_of_singletons,
        "isolated_fraction_total": isolated_fraction_total,
        "isolated_fraction_phenotype": isolated_fraction_phenotype,
        "mean_finite_cluster_size": mean_finite_cluster_size,
        "arithmetic_mean_without_largest": (
            arithmetic_mean_without_largest
        ),
    }


def calculate_size_counts(
    raw_data: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Count clusters by graph definition and size in one snapshot."""
    counts = (
        raw_data.groupby(GRAPH_KEYS + ["size"])
        .size()
        .rename("count")
        .reset_index()
    )
    return add_metadata(counts, metadata)


def process_raw_files(
    raw_files: list[Path],
    raw_output_path: Path,
    delta_t_fallback: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process raw cluster files and stream them into one parquet file."""
    snapshot_rows = []
    size_count_tables = []
    writer = None

    if raw_output_path.exists():
        raw_output_path.unlink()

    try:
        for index, filepath in enumerate(raw_files, start=1):
            metadata = parse_metadata(
                filepath,
                delta_t_fallback=delta_t_fallback,
            )
            raw_data = pd.read_csv(filepath, skipinitialspace=True)

            required_columns = set(GRAPH_KEYS + ["size"])
            missing = sorted(required_columns.difference(raw_data.columns))
            if missing:
                raise ValueError(
                    f"Missing cluster columns in {filepath}: {missing}"
                )

            for graph_values, graph_data in raw_data.groupby(
                GRAPH_KEYS,
                sort=True,
                dropna=False,
            ):
                phenotype, alignment, range_factor = graph_values
                snapshot_rows.append(
                    calculate_snapshot_statistics(
                        graph_data=graph_data,
                        metadata=metadata,
                        phenotype=str(phenotype),
                        alignment=str(alignment),
                        range_factor=float(range_factor),
                    )
                )

            size_count_tables.append(
                calculate_size_counts(
                    raw_data=raw_data,
                    metadata=metadata,
                )
            )

            raw_with_metadata = add_metadata(raw_data, metadata)
            table = pa.Table.from_pandas(
                raw_with_metadata,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    raw_output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            if index % 500 == 0 or index == len(raw_files):
                print(
                    f"Raw cluster files processed: "
                    f"{index}/{len(raw_files)}"
                )
    finally:
        if writer is not None:
            writer.close()

    snapshots = pd.DataFrame(snapshot_rows)

    if size_count_tables:
        size_counts = pd.concat(size_count_tables, ignore_index=True)
    else:
        size_counts = pd.DataFrame()

    return snapshots, size_counts


def process_summary_files(
    summary_files: list[Path],
    delta_t_fallback: float | None = None,
) -> pd.DataFrame:
    """Combine compact cluster summaries into one graph-resolved time series."""
    tables = []

    required_columns = {
        "phenotype",
        "alignment",
        "range_factor",
        "total_number_of_cells",
        "largest_cluster_size",
        "second_largest_cluster_size",
        "largest_fraction_population",
        "largest_fraction_system",
        "mean_without_largest",
        "finite_weighted_size",
    }

    for index, filepath in enumerate(summary_files, start=1):
        metadata = parse_metadata(
            filepath,
            delta_t_fallback=delta_t_fallback,
        )
        summary = pd.read_csv(filepath, skipinitialspace=True)

        missing = sorted(required_columns.difference(summary.columns))
        if missing:
            raise ValueError(
                f"Missing cluster-summary columns in {filepath}: {missing}"
            )

        # In the core output, total_number_of_cells is the number of cells
        # belonging to this phenotype/graph. It must NOT be summed across
        # graph definitions because the same cells appear in several graphs.
        n_phenotype = summary["total_number_of_cells"].astype(int)
        system_n = int(metadata["actual_N"])

        summary["system_number_of_cells"] = system_n
        summary["number_of_cells_phenotype"] = n_phenotype
        summary["fraction_phenotype"] = n_phenotype / system_n

        summary["S1"] = summary["largest_cluster_size"].fillna(0).astype(int)
        summary["S2"] = (
            summary["second_largest_cluster_size"].fillna(0).astype(int)
        )

        summary["S1_over_N"] = summary["S1"] / system_n
        summary["S2_over_N"] = summary["S2"] / system_n

        summary["S1_over_Nphenotype"] = np.where(
            n_phenotype > 0,
            summary["S1"] / n_phenotype,
            np.nan,
        )
        summary["S2_over_Nphenotype"] = np.where(
            n_phenotype > 0,
            summary["S2"] / n_phenotype,
            np.nan,
        )

        denominator = summary["S1"] + summary["S2"]
        summary["psi"] = np.where(
            denominator > 0,
            (summary["S1"] - summary["S2"]) / denominator,
            np.nan,
        )

        # Compatibility aliases used in the previous notebooks.
        summary["mean_finite_cluster_size"] = summary[
            "finite_weighted_size"
        ]
        summary["arithmetic_mean_without_largest"] = summary[
            "mean_without_largest"
        ]

        tables.append(add_metadata(summary, metadata))

        if index % 1000 == 0 or index == len(summary_files):
            print(
                f"Summary cluster files processed: "
                f"{index}/{len(summary_files)}"
            )

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


def merge_raw_observables_into_time_series(
    time_series: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Add raw-only observables at timesteps where raw clusters were saved."""
    if time_series.empty or snapshots.empty:
        return time_series

    merge_keys = SNAPSHOT_KEYS + GRAPH_KEYS
    raw_only_columns = merge_keys + [
        "number_of_singletons",
        "isolated_fraction_total",
        "isolated_fraction_phenotype",
    ]

    return time_series.merge(
        snapshots[raw_only_columns],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )


def validate_cluster_time_series(data: pd.DataFrame) -> None:
    """Basic graph-resolution and normalization checks."""
    if data.empty:
        return

    keys = SNAPSHOT_KEYS + GRAPH_KEYS
    duplicated = int(data.duplicated(subset=keys).sum())

    invalid_fraction = int(
        (
            data["fraction_phenotype"].notna()
            & ~data["fraction_phenotype"].between(0, 1)
        ).sum()
    )

    invalid_s1_population = int(
        (
            data["S1_over_Nphenotype"].notna()
            & ~data["S1_over_Nphenotype"].between(0, 1)
        ).sum()
    )

    # For a given phenotype and snapshot, the number of phenotype cells
    # should be independent of the graph definition.
    spread = (
        data.groupby(SNAPSHOT_KEYS + ["phenotype"])[
            "number_of_cells_phenotype"
        ]
        .agg(lambda values: values.max() - values.min())
    )
    inconsistent_phenotype_counts = int((spread != 0).sum())

    print("\nSanity checks:")
    print("Duplicated graph-snapshot rows:", duplicated)
    print("Invalid phenotype fractions:", invalid_fraction)
    print("Invalid S1 / N_phenotype values:", invalid_s1_population)
    print(
        "Phenotype snapshots with graph-dependent cell counts:",
        inconsistent_phenotype_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate graph-resolved cluster output files into parquet "
            "datasets."
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
            "<data_root>/processed/clusters"
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
        else data_root / "processed" / "clusters"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(data_root.rglob("cluster_summary_*.dat"))
    raw_files = sorted(data_root.rglob("cluster_sizes_*.dat"))

    print(f"Summary files found: {len(summary_files)}")
    print(f"Raw files found: {len(raw_files)}")

    if not summary_files and not raw_files:
        raise FileNotFoundError(
            f"No cluster files found below {data_root}"
        )

    raw_output_path = output_dir / "cluster_raw.parquet"

    if raw_files:
        snapshots, size_counts = process_raw_files(
            raw_files=raw_files,
            raw_output_path=raw_output_path,
            delta_t_fallback=args.delta_t,
        )
    else:
        snapshots = pd.DataFrame()
        size_counts = pd.DataFrame()

    time_series = process_summary_files(
        summary_files,
        delta_t_fallback=args.delta_t,
    )
    time_series = merge_raw_observables_into_time_series(
        time_series=time_series,
        snapshots=snapshots,
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
        "phenotype",
        "alignment",
        "range_factor",
    ]

    if not time_series.empty:
        time_series = time_series.sort_values(
            sort_columns
        ).reset_index(drop=True)
        validate_cluster_time_series(time_series)
        time_series.to_parquet(
            output_dir / "cluster_time_series.parquet",
            index=False,
        )

    if not snapshots.empty:
        snapshots = snapshots.sort_values(
            sort_columns
        ).reset_index(drop=True)
        snapshots.to_parquet(
            output_dir / "cluster_snapshots.parquet",
            index=False,
        )

    if not size_counts.empty:
        size_counts = size_counts.sort_values(
            sort_columns + ["size"]
        ).reset_index(drop=True)
        size_counts.to_parquet(
            output_dir / "cluster_size_counts.parquet",
            index=False,
        )

    print(f"Processed cluster data written to: {output_dir}")


if __name__ == "__main__":
    main()
